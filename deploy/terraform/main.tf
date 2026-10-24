// Ring Clip Agent demo on AWS: one Graviton instance running the playground (OpenCV 5 analysis results,
// Strands Decider 2B on CPU, Amazon Bedrock second looks) behind nginx, with the analysed clips in S3.
// By default the instance boots COOL's Graviton4 image and the pipeline uses COOL's OpenCV build.
//
//   terraform init && terraform apply      # build (about 10-15 minutes until the page answers)
//   terraform destroy                      # remove everything
//
// No SSH and no access keys on the instance: admin access is AWS Systems Manager Session Manager,
// and the instance role can call only Bedrock (Nova Lite / Micro), read its own bucket, and write logs.

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws    = { source = "hashicorp/aws", version = "~> 5.0" }
    random = { source = "hashicorp/random", version = "~> 3.6" }
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = { Project = var.project, ManagedBy = "terraform" }
  }
}

data "aws_caller_identity" "me" {}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
  filter {
    name   = "default-for-az"
    values = ["true"]
  }
}

data "aws_ami" "ubuntu_arm64" {
  most_recent = true
  owners      = ["099720109477"] // Canonical
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-arm64-server-*"]
  }
  filter {
    name   = "architecture"
    values = ["arm64"]
  }
}

// ---------------------------------------------------------------- the analysed clips

resource "random_id" "suffix" {
  byte_length = 4
}

resource "aws_s3_bucket" "samples" {
  bucket        = "${var.project}-samples-${random_id.suffix.hex}"
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "samples" {
  bucket                  = aws_s3_bucket.samples.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "samples" {
  bucket = aws_s3_bucket.samples.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

locals {
  samples_dir = "${path.module}/../../web/public/samples"
  // Only the published clips go up: never whatever else happens to be in the local results folder.
  published     = [for l in split("\n", file("${path.module}/../published_clips.txt")) : trimspace(l) if trimspace(l) != "" && !startswith(trimspace(l), "#")]
  sample_files  = toset(flatten([for c in local.published : ["${c}.mp4", "${c}.timeline.json"]]))
  content_types = { mp4 = "video/mp4", json = "application/json" }
}

resource "aws_s3_object" "sample" {
  for_each     = local.sample_files
  bucket       = aws_s3_bucket.samples.id
  key          = "samples/${each.value}"
  source       = "${local.samples_dir}/${each.value}"
  etag         = filemd5("${local.samples_dir}/${each.value}")
  content_type = local.content_types[regex("[^.]+$", each.value)]
}

// ---------------------------------------------------------------- permissions

resource "aws_iam_role" "app" {
  name = "${var.project}-instance"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Principal = { Service = "ec2.amazonaws.com" }, Action = "sts:AssumeRole" }]
  })
}

resource "aws_iam_role_policy" "app" {
  name = "${var.project}-least-privilege"
  role = aws_iam_role.app.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "BedrockNovaOnly"
        Effect   = "Allow"
        Action   = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"]
        Resource = [for m in var.bedrock_models : "arn:aws:bedrock:${var.region}::foundation-model/${m}"]
      },
      {
        Sid      = "ReadOwnSamples"
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:ListBucket"]
        Resource = [aws_s3_bucket.samples.arn, "${aws_s3_bucket.samples.arn}/*"]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.app.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy_attachment" "cloudwatch" {
  role       = aws_iam_role.app.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"
}

resource "aws_iam_instance_profile" "app" {
  name = "${var.project}-instance"
  role = aws_iam_role.app.name
}

// ---------------------------------------------------------------- network

resource "aws_security_group" "web" {
  name        = "${var.project}-web"
  description = "HTTP only; no SSH (admin through Session Manager)"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "web page"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = [var.allowed_cidr]
  }

  egress {
    description = "package installs, model download, Bedrock, S3, CloudWatch"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

// ---------------------------------------------------------------- logs

resource "aws_cloudwatch_log_group" "app" {
  name              = "/${var.project}"
  retention_in_days = 14
}

// ---------------------------------------------------------------- the instance

resource "aws_instance" "app" {
  ami                    = var.ami_id != "" ? var.ami_id : data.aws_ami.ubuntu_arm64.id
  instance_type          = var.instance_type
  subnet_id              = sort(data.aws_subnets.default.ids)[0]
  vpc_security_group_ids = [aws_security_group.web.id]
  iam_instance_profile   = aws_iam_instance_profile.app.name

  root_block_device {
    volume_size = var.disk_gb
    volume_type = "gp3"
    encrypted   = true
  }

  metadata_options {
    http_tokens   = "required" // IMDSv2 only
    http_endpoint = "enabled"
  }

  user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    region             = var.region
    repo_url           = var.repo_url
    repo_ref           = var.repo_ref
    bucket             = aws_s3_bucket.samples.id
    decider_checkpoint = var.decider_checkpoint
    vision_model       = "bedrock:${var.bedrock_models[0]}"
    log_group          = aws_cloudwatch_log_group.app.name
    use_cool           = var.use_cool
  })
  user_data_replace_on_change = true

  tags = { Name = var.project }

  depends_on = [aws_s3_object.sample]
}

resource "aws_eip" "app" {
  instance = aws_instance.app.id
  domain   = "vpc"
}

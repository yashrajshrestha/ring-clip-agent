variable "project" {
  description = "Name prefix for every resource."
  type        = string
  default     = "ring-clip-agent"
}

variable "region" {
  description = "AWS region (Bedrock Nova models must be available here)."
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "Graviton instance: Decider 2B needs about 5.5 GB of RAM on CPU, so 16 GB leaves headroom. COOL's Graviton4 image runs on c8g, m8g and r8g."
  type        = string
  default     = "t4g.xlarge"
}

variable "ami_id" {
  description = "Machine image. Empty: Canonical Ubuntu 24.04 arm64. For COOL (Cloud-Optimized OpenCV Library) on Graviton4, subscribe in AWS Marketplace and set ami-033e481a24f94c8cb with a c8g, m8g or r8g instance and use_cool = true."
  type        = string
  default     = ""
}

variable "use_cool" {
  description = "Run the pipeline on COOL's OpenCV build (/opt/cool) instead of the pip wheel. Needs the COOL image."
  type        = bool
  default     = false
}

variable "disk_gb" {
  description = "Root volume: OS, Python packages, ~5 GB of model weights, the clips."
  type        = number
  default     = 40
}

variable "repo_url" {
  description = "Public repository the instance clones."
  type        = string
  default     = "https://github.com/yashrajshrestha/ring-clip-agent.git"
}

variable "repo_ref" {
  description = "Branch or tag to deploy."
  type        = string
  default     = "main"
}

variable "decider_checkpoint" {
  description = "Strands Decider 2B checkpoint on Hugging Face."
  type        = string
  default     = "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"
}

variable "bedrock_models" {
  description = "Bedrock models the instance may call. The first is the vision second look."
  type        = list(string)
  default     = ["amazon.nova-lite-v1:0", "amazon.nova-micro-v1:0"]
}

variable "allowed_cidr" {
  description = "Who may open the web page."
  type        = string
  default     = "0.0.0.0/0"
}

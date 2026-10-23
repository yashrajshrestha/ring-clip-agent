output "url" {
  description = "The live demo."
  value       = "http://${aws_eip.app.public_ip}/"
}

output "instance_id" {
  value = aws_instance.app.id
}

output "samples_bucket" {
  value = aws_s3_bucket.samples.id
}

output "admin_shell" {
  description = "Open a shell on the instance (no SSH needed; requires the Session Manager plugin)."
  value       = "aws ssm start-session --region ${var.region} --target ${aws_instance.app.id}"
}

output "logs" {
  description = "Follow the app logs in CloudWatch."
  value       = "aws logs tail /${var.project} --region ${var.region} --follow"
}

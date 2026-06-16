output "ec2_instance_id" {
  value = aws_instance.training.id
}

output "ec2_public_ip" {
  value = aws_instance.training.public_ip
}

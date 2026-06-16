# Get default VPC
data "aws_vpc" "default" {
  default = true
}

# Get default subnets
data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# Security Group
resource "aws_security_group" "ec2" {
  name        = "${var.project_name}-ec2-sg"
  description = "Allow SSH and outbound"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# IAM Role for ECR
resource "aws_iam_role" "ec2_role" {
  name = "${var.project_name}-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ecr_power" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser"
}

resource "aws_iam_role_policy_attachment" "s3_full" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

resource "aws_iam_instance_profile" "ec2_profile" {
  name = "${var.project_name}-ec2-profile"
  role = aws_iam_role.ec2_role.name
}

# Deep Learning AMI
data "aws_ami" "dlami" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Amazon Linux 2023) *"]
  }
}

# EC2 Instance (On-Demand, Spot limits exceeded in region)
resource "aws_instance" "training" {
  ami                    = data.aws_ami.dlami.id
  instance_type          = var.instance_type
  key_name               = var.ssh_key_name
  iam_instance_profile   = aws_iam_instance_profile.ec2_profile.name
  vpc_security_group_ids = [aws_security_group.ec2.id]

  root_block_device {
    volume_size = 100
    volume_type = "gp3"
  }

  user_data = <<-EOF
              #!/bin/bash
              set -e
              echo "=== EC2 Bootstrap ===" >> /var/log/user-data.log

              # Update and install Docker
              dnf update -y >> /var/log/user-data.log 2>&1
              dnf install -y docker >> /var/log/user-data.log 2>&1
              systemctl start docker >> /var/log/user-data.log 2>&1
              systemctl enable docker >> /var/log/user-data.log 2>&1

              # Download and extract code
              echo "Downloading code from S3..." >> /var/log/user-data.log
              WORK_DIR="/home/ec2-user/tfm-slm"
              mkdir -p "$WORK_DIR"
              cd "$WORK_DIR"
              aws s3 cp s3://tfm-slm-code/tfm-slm-code.zip . >> /var/log/user-data.log 2>&1
              unzip -qo tfm-slm-code.zip >> /var/log/user-data.log 2>&1

              # Run build-and-train script
              echo "Starting build and training..." >> /var/log/user-data.log
              chmod +x build-and-train.sh
              export MODE=${var.deployment_mode}
              export AWS_REGION=${var.aws_region}
              bash build-and-train.sh >> /var/log/user-data.log 2>&1 || echo "Build/train failed" >> /var/log/user-data.log

              echo "=== EC2 Bootstrap Complete ===" >> /var/log/user-data.log
              EOF

  tags = {
    Name = var.project_name
  }
}

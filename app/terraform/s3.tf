resource "aws_s3_bucket" "checkpoints" {
  bucket        = "tfm-slm-checkpoints"
  force_destroy = true

  tags = {
    Name        = "tfm-slm-checkpoints"
    Environment = "production"
  }
}

resource "aws_s3_bucket_versioning" "checkpoints" {
  bucket = aws_s3_bucket.checkpoints.id
  versioning_configuration {
    status = "Enabled"
  }
}

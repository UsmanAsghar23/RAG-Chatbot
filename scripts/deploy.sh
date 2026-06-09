#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TERRAFORM_DIR="$ROOT_DIR/infra/terraform"
AWS_REGION="${AWS_REGION:-us-east-1}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

cd "$ROOT_DIR"

ECR_URL="$(terraform -chdir="$TERRAFORM_DIR" output -raw ecr_repository_url)"
CLUSTER="$(terraform -chdir="$TERRAFORM_DIR" output -raw ecs_cluster_name)"
SERVICE="$(terraform -chdir="$TERRAFORM_DIR" output -raw ecs_service_name)"

echo "Building Docker image..."
docker build -t rag-api:"$IMAGE_TAG" .

echo "Logging in to ECR..."
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "${ECR_URL%/*}"

echo "Pushing $ECR_URL:$IMAGE_TAG"
docker tag rag-api:"$IMAGE_TAG" "$ECR_URL:$IMAGE_TAG"
docker push "$ECR_URL:$IMAGE_TAG"

echo "Rolling ECS service $SERVICE on cluster $CLUSTER"
aws ecs update-service \
  --region "$AWS_REGION" \
  --cluster "$CLUSTER" \
  --service "$SERVICE" \
  --force-new-deployment \
  --output text >/dev/null

echo "Deploy triggered. Check ECS service events and CloudWatch logs."
ALB_URL="$(terraform -chdir="$TERRAFORM_DIR" output -raw alb_url)"
echo "Health check: curl $ALB_URL/health"

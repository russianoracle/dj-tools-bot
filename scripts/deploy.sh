#!/bin/bash
# Deployment script for Yandex Cloud
# Usage: ./scripts/deploy.sh [init|plan|apply|destroy]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TERRAFORM_DIR="$PROJECT_ROOT/terraform"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check Terraform
    if ! command -v terraform &> /dev/null; then
        log_error "Terraform is not installed. Install it with: brew install terraform"
        exit 1
    fi

    # Check Yandex Cloud CLI
    if ! command -v yc &> /dev/null; then
        log_error "Yandex Cloud CLI is not installed. Install it from: https://cloud.yandex.ru/docs/cli/quickstart"
        exit 1
    fi

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed."
        exit 1
    fi

    # Check terraform.tfvars exists
    if [ ! -f "$TERRAFORM_DIR/terraform.tfvars" ]; then
        log_error "terraform.tfvars not found!"
        log_info "Copy terraform.tfvars.example to terraform.tfvars and fill in your values:"
        log_info "  cp $TERRAFORM_DIR/terraform.tfvars.example $TERRAFORM_DIR/terraform.tfvars"
        exit 1
    fi

    log_info "All prerequisites met!"
}

terraform_init() {
    log_info "Initializing Terraform..."
    cd "$TERRAFORM_DIR"
    terraform init
}

terraform_plan() {
    log_info "Planning infrastructure changes..."
    cd "$TERRAFORM_DIR"
    terraform plan -out=tfplan
}

terraform_apply() {
    log_info "Applying infrastructure changes..."
    cd "$TERRAFORM_DIR"

    if [ -f tfplan ]; then
        terraform apply tfplan
        rm tfplan
    else
        terraform apply
    fi

    # Get outputs
    log_info "Deployment complete! Getting outputs..."
    terraform output
}

terraform_destroy() {
    log_warn "This will DESTROY all infrastructure!"
    read -p "Are you sure? (yes/no): " confirm
    if [ "$confirm" = "yes" ]; then
        cd "$TERRAFORM_DIR"
        terraform destroy
    else
        log_info "Destroy cancelled."
    fi
}

build_and_push() {
    log_info "Building and pushing Docker image..."

    cd "$PROJECT_ROOT"

    # Get registry info from terraform output
    cd "$TERRAFORM_DIR"
    REGISTRY_ID=$(terraform output -raw registry_id 2>/dev/null || echo "")

    if [ -z "$REGISTRY_ID" ]; then
        log_error "Registry ID not found. Run terraform apply first."
        exit 1
    fi

    IMAGE_TAG="cr.yandex/$REGISTRY_ID/mood-classifier:latest"

    cd "$PROJECT_ROOT"

    # Build image
    log_info "Building image: $IMAGE_TAG"
    docker build -t "$IMAGE_TAG" .

    # Login to registry
    log_info "Logging in to Yandex Container Registry..."
    yc container registry configure-docker

    # Push image
    log_info "Pushing image..."
    docker push "$IMAGE_TAG"

    log_info "Image pushed successfully: $IMAGE_TAG"
}

show_status() {
    log_info "Infrastructure status:"
    cd "$TERRAFORM_DIR"
    terraform show
}

show_outputs() {
    log_info "Terraform outputs:"
    cd "$TERRAFORM_DIR"
    terraform output
}

case "${1:-help}" in
    init)
        check_prerequisites
        terraform_init
        ;;
    plan)
        check_prerequisites
        terraform_plan
        ;;
    apply)
        check_prerequisites
        terraform_apply
        ;;
    destroy)
        check_prerequisites
        terraform_destroy
        ;;
    build)
        check_prerequisites
        build_and_push
        ;;
    status)
        show_status
        ;;
    outputs)
        show_outputs
        ;;
    full)
        check_prerequisites
        terraform_init
        terraform_plan
        terraform_apply
        build_and_push
        ;;
    help|*)
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  init     - Initialize Terraform"
        echo "  plan     - Plan infrastructure changes"
        echo "  apply    - Apply infrastructure changes"
        echo "  destroy  - Destroy all infrastructure"
        echo "  build    - Build and push Docker image"
        echo "  status   - Show infrastructure status"
        echo "  outputs  - Show Terraform outputs"
        echo "  full     - Full deployment (init + plan + apply + build)"
        echo ""
        echo "First time setup:"
        echo "  1. cp terraform/terraform.tfvars.example terraform/terraform.tfvars"
        echo "  2. Edit terraform.tfvars with your values"
        echo "  3. ./scripts/deploy.sh full"
        ;;
esac

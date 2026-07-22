# Infrastructure

Terraform for running the API on Google Cloud Run.

It creates:

- an Artifact Registry repository for the API image, with a cleanup policy that keeps the two newest images
- a Cloud Run service that scales to zero, capped at 2 instances, publicly invokable
- a runtime service account with no extra permissions
- Workload Identity Federation so GitHub Actions can deploy without a stored key, restricted to this repository

## First-time setup

1. Create a Google Cloud project and attach a billing account.
2. Authenticate locally: `gcloud auth application-default login`.
3. Copy `terraform.tfvars.example` to `terraform.tfvars` and fill in the project id.
4. Apply:

   ```bash
   terraform init
   terraform apply
   ```

5. Copy the outputs into GitHub repository variables (Settings → Secrets and variables → Actions → Variables):
   `GCP_PROJECT_ID`, `GCP_REGION`, `AR_REPOSITORY`, `CLOUD_RUN_SERVICE`,
   `GCP_WORKLOAD_IDENTITY_PROVIDER`, `GCP_DEPLOY_SERVICE_ACCOUNT`.
6. Publish the model as a GitHub release tagged `model-v1` with `artifacts/model.pt` and `artifacts/movies.parquet` attached.
7. Run the **Deploy API** workflow, or push a `v*` tag.

Until step 7 the service shows Google's placeholder container.

## Cost

The service scales to zero and bills CPU only while handling requests, so a low-traffic demo is expected to stay inside Cloud Run's monthly free allowance. Image storage in Artifact Registry is billed per GB above a small free allowance. Set a budget alert on the billing account so any charge is noticed early.

State is stored locally by default. `terraform.tfstate` and `terraform.tfvars` are git-ignored.

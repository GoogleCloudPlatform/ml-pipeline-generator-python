len=$(gcloud container clusters describe cluster-6 | grep 'cloud-platform')
if [[ $len == *"cloud-platform"* ]]; then
  echo "It's there!"
fi
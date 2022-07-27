


delete_unused_files() {
  # Delete the unnecessary files in GZ2 image directory
  echo "Deleting unused files for Image Dataset"
  rm -r ../../data/images_gz2/images/results_public.txt
  rm -r ../../data/images_gz2/images/results.txt
  rm -r ../../data/images_gz2/images/z
  local exit_code=$?
  handle_exit_code ${exit_code} "Delete failed."
  echo "Successful!"
}

configure() {
  delete_unused_files
}

configure

exit 0
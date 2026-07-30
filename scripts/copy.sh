cd /home/airlab/SousVide/gsplats/capture/aerialarena_square && find . -type d | while read -r dir; do
  mkdir -p "/home/airlab/SousVide/gsplats/capture/aerialarena_square_subset/$dir"
  find "$dir" -maxdepth 1 -type f | sort -V | head -n 40 | while read -r file; do
    cp "$file" "/home/airlab/SousVide/gsplats/capture/aerialarena_square_subset/$file"
  done
done

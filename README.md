Biar bisa jalan

Download datasetnya disini :
https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs
, download yang zip
, lalu rename datasetnya menjadi dataset_spotify_tracks.csv
, file asli 400 MB (x_x) 1.2+ juta lagu spotify.

Ribet banget infokan jasa kompres dataset tapi tetap csv wkwk.

Ok Google "Can you compress a dataset file without compressing" 

Buat API spotify di:
developer.spotify.com
, ambil client id dan client secret setelah membuat API
, buat folder .streamlit, lalu buat file secrets.toml untuk menaruh client id dan client secret.

cek_koneksi.py masih belum bagus jadi disarankan langsung cek di selerafy.py nya

Playlist harus public dan tidak ada lagu impor dari luar (nggak dari spotify)
, disarankan menggunakan playlist buatan sendiri atau orang lain jangan buatan spotify.

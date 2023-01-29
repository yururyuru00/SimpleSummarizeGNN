IFS_BACKUP=$IFS
IFS=$'\n'

ary=(
     "python3 debug/plot_t_sne.py t-sne/gat_pubmed_l2.npy t-sne/y_true.npy t-sne/gat_pubmed_l2.png"
     "python3 debug/plot_t_sne.py t-sne/gat_pubmed_l9.npy t-sne/y_true.npy t-sne/gat_pubmed_l9.png"
     "python3 debug/plot_t_sne.py t-sne/twingat_pubmed_l2.npy t-sne/y_true.npy t-sne/twingat_pubmed_l2.png"
     "python3 debug/plot_t_sne.py t-sne/twingat_pubmed_l9.npy t-sne/y_true.npy t-sne/twingat_pubmed_l9.png"
     "python3 debug/plot_t_sne.py t-sne/summarizegat_pubmed_l2.npy t-sne/y_true.npy t-sne/summarizegat_pubmed_l2.png"
     "python3 debug/plot_t_sne.py t-sne/summarizegat_pubmed_l9.npy t-sne/y_true.npy t-sne/summarizegat_pubmed_l9.png"
     )

for STR in ${ary[@]}
do
    eval "${STR}"
done

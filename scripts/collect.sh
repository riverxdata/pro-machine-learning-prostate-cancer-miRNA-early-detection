#!bin/bash
# Download data/ Can do alternative option using GEOparse Python, or GEO2R in R


for i in `cat $1 |sed "1d"`
do
echo "Downloading on $i"
data_lake=${i:0:-3}
echo "Retrieved from https://ftp.ncbi.nlm.nih.gov/geo/series/${data_lake}nnn/${i}/matrix/${i}_series_matrix.txt.gz ..."
wget https://ftp.ncbi.nlm.nih.gov/geo/series/${data_lake}nnn/${i}/matrix/${i}_series_matrix.txt.gz
done


# Process data
for i in `find *.gz`
do
k=`basename -- $i .txt.gz`
echo "Processing ... ${k}"
# All infor
zcat $i > ${k}.txt
# Clinical information
zcat $i|grep "ID" > clin_${k}.txt
zcat $i|grep "charac" >> clin_${k}.txt
# Processed data
start_line=`zcat $i|nl|grep "series_matrix_table_begin"|awk '{print $1}'`
start_line=$((start_line+1))
zcat $i|awk "NR > ${start_line}" > data_${k}.txt
done   
rm *.gz

python3 process.py ./ $3

# Save data
rm -rf $2
mkdir $2
mv *GSE* $2





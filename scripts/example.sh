#1.Prepare the data
# Here, provide the design file only one columns where the data are processed similar, if not this scripts 
# does not work
# Take a look at the input from design.tsv

# data
# GSE112264
# GSE113486

#  bash collect.sh "the directory for file input" "the output directory" "the single word contain that target predict"
bash collect.sh ../data/design.tsv ../example_data disease

#2. To run machine learning, follow the tutorial on python script example.py
example.py

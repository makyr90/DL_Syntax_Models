#Run python programm with command line arguments

python3 biaffine_parser.py --predict \
--model el_model.model \
--params el_model.paramsdr \
--outdir sample/ \
--test sample/el-ud-dev.conllu \
--output el-ud-dev.conllu.pred

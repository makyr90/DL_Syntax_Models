#Run python programm with command line arguments

python3 tagger.py --predict \
--model el_model.model \
--params el_model.paramsdr \
--outdir sample/ \
--test sample/el-ud-test.conllu \
--output el-ud-test.conllu.pred

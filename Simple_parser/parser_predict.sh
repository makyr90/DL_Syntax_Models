#Run python programm with command line arguments

python3 parser.py --predict \
--predict-batch 16 \
--model el_model.model \
--params el_model.paramsdr \
--outdir data/ \
--test data/el.conllu \
--output el-ud-test.conllu.pred

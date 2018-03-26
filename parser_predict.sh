#Run python programm with command line arguments

python3 parser.py --predict \
--model el_model.model \
--params el_model.paramsdr \
--predict-batch 32 \
--outdir data/ \
--test data/el.conllu \
--output el-ud-test.conllu.pred

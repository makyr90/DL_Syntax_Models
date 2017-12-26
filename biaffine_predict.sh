#Run python programm with command line arguments

python3 biaffine_parser.py --predict \
--model el_model.model \
--params el_model.paramsdr \
--outdir data/ \
--test data/el-ud-test_predicted_posxpos.conllu \
--output el-ud-test.conllu.pred

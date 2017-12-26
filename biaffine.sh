#Run python programm with command line arguments
#Also keep a log file for the console output

stdbuf -oL python3 -u biaffine_parser.py --dynet-seed 123456789 \
--epochs 4000 \
--predict-batch 32 \
--batch-tokens 3000 \
--dropout 0.33 \
--extrn sample/gr_fasttext.npy \
--extrn-voc sample/gr_fasttext.vocab \
--wembedding 100 \
--cembedding 100 \
--posembedding 100 \
--lstmlayers 3 \
--lstmdims 200 \
--model el_model.model \
--params el_model.paramsdr \
--outdir sample/ \
--train data/el-ud-train.conllu \
--dev data/el-ud-dev.conllu |  while IFS= read -r line
do
tee -a model_log.txt
done

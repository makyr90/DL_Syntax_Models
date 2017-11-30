#Run python programm with command line arguments
#Also keep a log file for the console output and shutdown when finish

stdbuf -oL python3 -u biaffine_parser.py --dynet-seed 123456789 \
--epochs 4000 \
--predict-batch 32 \
--batch-tokens 3000 \
--dropout 0.33 \
--extrn sample/gr_GloVe.npy \
--extrn-voc sample/gr_GloVe.vocab \
--wembedding 100 \
--cembedding 100 \
--posembedding 100 \
--lstmlayers 3 \
--lstmdims 200 \
--model el_model.model \
--params el_model.paramsdr \
--outdir sample/ \
--train sample/el-ud-train.conllu \
--dev sample/el-ud-dev.conllu |  while IFS= read -r line
do
tee -a model_log.txt
done

# sudo shutdown -P +2

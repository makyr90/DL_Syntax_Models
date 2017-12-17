# coding=utf-8
from optparse import OptionParser
import pickle, utils, learner, os, sys, os.path, time, random





if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train", dest="conll_train", help="Path to annotated CONLL train file", metavar="FILE", default="N/A")
    parser.add_option("--dev", dest="conll_dev", help="Path to annotated CONLL dev file", metavar="FILE", default="N/A")
    parser.add_option("--test", dest="conll_test", help="Path to CONLL test file", metavar="FILE", default="N/A")
    parser.add_option("--output", dest="conll_test_output", help="File name for predicted output", metavar="FILE", default="N/A")
    parser.add_option("--outdir", type="string", dest="output", default="results")
    parser.add_option("--extrn", dest="external_embedding", help="External embeddings", metavar="FILE")
    parser.add_option("--extrn-voc", dest="external_embedding_voc", help="External embeddings vocabulary", metavar="FILE",default="N/A")
    parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="model.params")
    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE", default="model")
    parser.add_option("--dropout", dest="dropout", help="dropout value", metavar="FILE", default=0)
    parser.add_option("--wembedding", type="int", dest="wembedding_dims", default=128)
    parser.add_option("--cembedding", type="int", dest="cembedding_dims", default=64)
    parser.add_option("--batch-tokens", type="int", dest="batch_tokens", default=5000)
    parser.add_option("--predict-batch", type="int", dest="pred_batch_size", default=1)
    parser.add_option("--epochs", type="int", dest="epochs", default=30)
    parser.add_option("--lr", type="float", dest="learning_rate", default=2e-3)
    parser.add_option("--lstmlayers", type="int", dest="lstm_layers", default=2)
    parser.add_option("--lstmdims", type="int", dest="lstm_dims", default=128)
    parser.add_option("--predict", action="store_true", dest="predictFlag", default=False)
    parser.add_option("--dynet-seed", type="int", dest="seed", default=0)
    parser.add_option("--dynet-mem", type="int", dest="mem", default=0)
    parser.add_option("--last-epoch", type="int", dest="last_epoch", default=0)
    parser.add_option("--highest-score", type="float", dest="highest_score", default=0.0)
    parser.add_option("--dynet-gpus")
    parser.add_option("--dynet-devices")



    (options, args) = parser.parse_args()

    print('Using external embedding:', options.external_embedding)

    if options.predictFlag:
        with open(os.path.join(options.output, options.params), 'rb') as paramsfp:
            words, w2i, c2i, pos, xpos, stored_opt = pickle.load(paramsfp)

        ext_words_train = utils.ext_vocab(stored_opt.conll_train,stored_opt.external_embedding_voc)
        ext_words_test = utils.ext_vocab(options.conll_test,stored_opt.external_embedding_voc)

        print('Loading pre-trained  model')
        tagger = learner.Affine_tagger(words, pos, xpos,  w2i, c2i, ext_words_train, ext_words_test, stored_opt)
        tagger.Load(os.path.join(options.output, os.path.basename(options.model)))

        tespath = os.path.join(options.output, options.conll_test_output)
        print('Predicting  POS XPOS tags')
        ts = time.time()
        test_res = list(tagger.Predict(options.conll_test,True))
        te = time.time()
        print('Finished in', te-ts, 'seconds.')
        utils.write_conll(tespath, test_res)


    else:

        ext_words_train = utils.ext_vocab(options.conll_train,options.external_embedding_voc)
        ext_words_dev = utils.ext_vocab(options.conll_dev,options.external_embedding_voc)

        if (options.last_epoch == 0):

            print('Extracting vocabulary')
            words, w2i, c2i, pos, xpos = utils.vocab(options.conll_train)

            with open(os.path.join(options.output, options.params), 'wb') as paramsfp:
                pickle.dump((words, w2i, c2i, pos, xpos, options), paramsfp)

            print('Initializing  model')
            tagger = learner.Affine_tagger(words, pos, xpos,  w2i, c2i, ext_words_train, ext_words_dev, options)

            with open("Results.txt", "a") as results:
                results.write("T_Step\tPOS\tXPOS\n")
        else:

            with open(os.path.join(options.output, options.params), 'rb') as paramsfp:
                words, w2i, c2i, pos, xpos, stored_opt = pickle.load(paramsfp)


            stored_opt.external_embedding = options.external_embedding

            print('Loading  model')

            tagger = learner.Affine_tagger(words, pos, xpos,  w2i, c2i, ext_words_train, ext_words_dev, stored_opt)
            tagger.Load(os.path.join(options.output, os.path.basename(options.model)))



        sentences,train_batches = utils.batch_train_data(options.conll_train,c2i,options.batch_tokens)
        batches = len(train_batches)
        highestScore = options.highest_score
        tsId = options.last_epoch*batches
        for epoch in range(options.last_epoch,options.epochs):
            random.shuffle(train_batches)
            for idx,mini_batch in enumerate(train_batches):
                t_step = (epoch * batches) +idx+1
                if (t_step%5000 == 0):
                    tagger.Train(sentences,mini_batch,t_step,True)
                else:
                    tagger.Train(sentences,mini_batch,t_step)


                if (t_step <=1000):
                    if (t_step%100== 0):
                        print("Save Model...")
                        tagger.Save(os.path.join(options.output, os.path.basename(options.model)))

                else:
                    print("Performance on Dev data")
                    start = time.time()
                    devPredSents = tagger.Predict(options.conll_dev)
                    count = 0
                    posCount = 0
                    xposCount = 0

                    for idSent, devSent in enumerate(devPredSents):
                        conll_devSent = [entry for entry in devSent if isinstance(entry, utils.ConllEntry)]

                        for entry in conll_devSent:
                            if entry.id <= 0:
                                continue

                            if entry.pos == entry.pred_pos:
                                posCount += 1
                            if entry.xpos == entry.pred_xpos:
                                xposCount += 1
                            count += 1

                    print("Finish predictions on dev data in %.2fs" %  (time.time() - start))
                    print("---\nPOS accuracy:\t%.2f" % (float(posCount) * 100 / count))
                    print("XPOS accuracy:\t%.2f" % (float(xposCount) * 100 / count))
                    with open("Results.txt", "a") as results:
                        results.write(str(t_step)+"\t"+str(round((float(posCount) * 100 / count),2))+"\t"+str(round((float(xposCount) * 100 / count),2))+"\n")



                    score = ((float(posCount) * 100 / count) + (float(xposCount) * 100 / count))/2
                    if score >= highestScore:
                        tagger.Save(os.path.join(options.output, os.path.basename(options.model)))
                        highestScore = score
                        tsId = t_step

                    print("Highest avg POS/XPOS(@Dev): %.2f at trainning step %d" % (highestScore,tsId))

                    if ((t_step - tsId) > 5000):
                        print("Model trainning finish..")
                        print("Model  didn't improve during the last 5000 trainning steps")
                        sys.exit()

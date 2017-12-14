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
    parser.add_option("--posembedding", type="int", dest="posembedding_dims", default=64)
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
            words, w2i, c2i, pos, xpos, rels, stored_opt = pickle.load(paramsfp)

        stored_opt.external_embedding = options.external_embedding
        ext_words_train = utils.ext_vocab(options.conll_train,options.external_embedding_voc)
        ext_words_test = utils.ext_vocab(options.conll_test,options.external_embedding_voc)

        print('Loading pre-trained  model')
        biaf_parser = learner.biAffine_parser(words, pos, xpos, rels, w2i, c2i, ext_words_train, ext_words_test, stored_opt)
        biaf_parser.Load(options.model)

        tespath = os.path.join(options.output, options.conll_test_output)
        print('Predicting  parsing dependencies')
        ts = time.time()
        test_res = list(biaf_parser.Predict(options.conll_test))
        te = time.time()
        print('Finished in', te-ts, 'seconds.')
        utils.write_conll(tespath, test_res)


    else:

        ext_words_train = utils.ext_vocab(options.conll_train,options.external_embedding_voc)
        ext_words_dev = utils.ext_vocab(options.conll_dev,options.external_embedding_voc)

        if (options.last_epoch == 0):

            print('Extracting vocabulary')
            words, w2i, c2i, pos, xpos, rels = utils.vocab(options.conll_train)

            with open(os.path.join(options.output, options.params), 'wb') as paramsfp:
                pickle.dump((words, w2i, c2i, pos, xpos, rels, options), paramsfp)

            print('Initializing  model')
            biaf_parser = learner.biAffine_parser(words, pos, xpos, rels, w2i, c2i, ext_words_train, ext_words_dev, options)

            with open("Results.txt", "a") as results:
                results.write("T_Step\tUAS\tLAS\n")
        else:

            with open(os.path.join(options.output, options.params), 'rb') as paramsfp:
                words, w2i, c2i, pos, xpos, rels, stored_opt = pickle.load(paramsfp)


            stored_opt.external_embedding = options.external_embedding

            print('Loading  model')

            biaf_parser = learner.biAffine_parser(words, pos, xpos, rels, w2i, c2i, ext_words_train, ext_words_dev, stored_opt)
            biaf_parser.Load(os.path.join(options.output, os.path.basename(options.model)))



        sentences,train_batches = utils.batch_train_data(options.conll_train,c2i,options.batch_tokens)
        batches = len(train_batches)
        highestScore = options.highest_score
        tsId = options.last_epoch*batches
        for epoch in range(options.last_epoch,options.epochs):
            random.shuffle(train_batches)
            for idx,mini_batch in enumerate(train_batches):
                t_step = (epoch * batches) +idx+1
                if (t_step%5000 == 0):
                    biaf_parser.Train(sentences,mini_batch,t_step,True)
                else:
                    biaf_parser.Train(sentences,mini_batch,t_step)


                if (t_step <=1000):
                    if (t_step%100== 0):
                        print("Save Model...")
                        biaf_parser.Save(os.path.join(options.output, os.path.basename(options.model)))

                else:
                    print("Performance on Dev data")
                    start = time.time()
                    devPredSents = biaf_parser.Predict(options.conll_dev)
                    count = 0
                    lasCount = 0
                    uasCount = 0

                    for idSent, devSent in enumerate(devPredSents):
                        conll_devSent = [entry for entry in devSent if isinstance(entry, utils.ConllEntry)]

                        for entry in conll_devSent:
                            if entry.id <= 0:
                                continue

                            if entry.parent_id == entry.pred_parent_id:
                                uasCount += 1
                                if entry.pred_relation == entry.relation:
                                    lasCount += 1
                            count += 1

                    print("Finish predictions on dev data in %.2fs" %  (time.time() - start))
                    print("---\nUAS accuracy:\t%.2f" % (float(uasCount) * 100 / count))
                    print("LAS accuracy:\t%.2f" % (float(lasCount) * 100 / count))
                    with open("Results.txt", "a") as results:
                        results.write(str(t_step)+"\t"+str(round((float(uasCount) * 100 / count),2))+"\t"+str(round((float(lasCount) * 100 / count),2))+"\n")



                    score = (float(lasCount) * 100 / count)
                    if score >= highestScore:
                        biaf_parser.Save(os.path.join(options.output, os.path.basename(options.model)))
                        highestScore = score
                        tsId = t_step

                    print("Highest LAS(@Dev): %.2f at trainning step %d" % (highestScore,tsId))

                    if ((t_step - tsId) > 5000):
                        print("Model trainning finish..")
                        print("Model  didn't improve during the last 5000 trainning steps")
                        sys.exit()

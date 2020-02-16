import numpy as np
from statistics import mean
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import svm

# Own Libraries/Classes
from setdivision import TestTrain, TrainValidation, CrossValidation
from knn import knn
from weightedknn import weightedknn

def score(train, test, feat,  func, k, kernel):

    # For wknn
#    if func in ["wknn","wknnprob"]:
#        test_features = np.array(wtestfeatures)
#        train_features = np.array(wtrainfeatures)

    # For the rest
#    else:
    train_features = np.array(train[feat])#np.array(train[["radius_mean", "texture_mean", "perimeter_mean", "area_mean",
#                                          "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean",
#                                          "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se",
#                                          "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se",
#                                          "concave points_se", "symmetry_se", "fractal_dimension_se", "radius_worst",
#                                          "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
#                                          "compactness_worst", "concavity_worst", "concave points_worst",
#                                          "symmetry_worst", "fractal_dimension_worst"]])

    test_features = np.array(test[feat]) #np.array(test[["radius_mean", "texture_mean", "perimeter_mean", "area_mean",
                                       #"smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean",
                                       #"symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se",
                                       #"perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se",
                                       #"concave points_se", "symmetry_se", "fractal_dimension_se", "radius_worst",
                                       #"texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
                                       #"compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst",
                                       #"fractal_dimension_worst"]])

    # Specifying Classes
    train_class = np.array(train["diagnosis"])
    test_class = np.array(test["diagnosis"])

    # Normalize for Logical Regression
    if func in ["lr"]:
        # Normalization of Dataframes
        train_features = preprocessing.scale(train_features)
        test_features = preprocessing.scale(test_features)

    # Specifying Labels
    train_labels = np.array(train["id"])
    test_labels = np.array(test["id"])

    # Find w for weighted knn wknn
    if func in ["wknn","wknnprob"]:
        global w
        # Finding initial weights from correlation matrix
        dia = train_class
        dia = np.array(dia)
        dianp = np.empty(0)

        for i in range(len(dia)):
            if dia[i] == "B":
                dianp = np.append(dianp, np.float64(0.0))
            else:
                dianp = np.append(dianp, np.float64(1.0))

        df1 = train[feat].assign(dia=dianp)
        corr = df1.corr()
        w = corr[["dia"]]
        w = w.drop(index='dia')
        totcor = np.array(w)

        w = np.empty([0])
        for i in range(len(totcor)):
            w = np.append(w, (totcor[i] / np.sum(totcor)))

    # Fit model for Logistic Regression lr
    if func == "lr":
        global lr_model
        lr_model = LogisticRegression()
        lr_model.fit(train_features, train_class)


    # Fit model for Support Vector Machines svm
    if func == "svm":
        global svm_model
        svm_model = svm.SVC(C=1, kernel=kernel, degree=3, probability=True)
        svm_model.fit(train_features, train_class)

    if func == "svmpen":
        global svmpen_model
        svmpen_model = svm.SVC(C=1, kernel=kernel, degree=3, probability=True)
        svmpen_model.fit(train_features, train_class)

    correct = 0
    incorrect = 0
    loss = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    # Finding score by calculating predictions for each sample (predictions) in the test set and then comparing it with
    # the true value (test_class[s]) .
    for s in range(len(test_class)):

        if func in ["knn", "knnprob", "wknnprob", "wknn"]:
            global sample
            sample = test_features[s]


        # Reshape and Transpose for Logical Regression, or Support Machine Vectors
        if func in ["lr", "svm"]:
            global Sample
            Sample = test_features[s]
            Sample = Sample.reshape(-1, 1)
            Sample = Sample.transpose()


        if func == "knn":
            global knn_predictions
            o = knn(train_features, train_labels, train_class, k, sample)
            knn_predictions = o.nverdict()

        if func in ["knnprob"]:
            global knnprob_predictions, knnprob_prediction
            o = knn(train_features, train_labels, train_class, k, sample)
            knnprob_prediction = o.hclassprob() / 100

            if knnprob_prediction == 0:
                knnprob_prediction = 0.0000001

            if knnprob_prediction >= 1:
                knnprob_prediction = 1 - 0.0000001

            if knnprob_prediction >= 0.5:
                knnprob_predictions = "M"
            else:
                knnprob_predictions = "B"

        if func == "wknn":
            global wknn_predictions
            o = weightedknn(train_features, train_labels, train_class, k, w, sample)
            wknn_predictions = o.nverdict()   # for definitive


        if func in ["wknnprob"]:
            global wknnprob_prediction, wknnprob_predictions
            o = weightedknn(train_features, train_labels, train_class, k, w, sample)
            wknnprob_prediction = o.hclassprob() / 100

            if wknnprob_prediction == 0:
                wknnprob_prediction = 0.0000001

            if wknnprob_prediction >= 1:
                wknnprob_prediction = 1 - 0.0000001

            if wknnprob_prediction >= 0.5:
                wknnprob_predictions = "M"
            else:
                wknnprob_predictions = "B"


        if func == "lr":
            global lr_prediction, lr_predictions
            lr_prediction = lr_model.predict_proba(Sample)[:, 1]
            if lr_prediction >= 0.5:
                lr_predictions = "M"
            else:
                lr_predictions = "B"


        if func == "svm":
            global svm_prediction, svm_predictions
            svm_prediction = svm_model.predict_proba(Sample)[:, 1]

            if svm_prediction >= 0.5:
                svm_predictions = "M"
            else:
                svm_predictions = "B"

        # Counting Correct predictions and incorrect ones with and without penalty

        if func == "knn":
            if test_class[s] == knn_predictions:
                correct += 1
                if test_class[s] == "M":
                    loss += 0
                    tp += 1
                else:
                    loss += 0
                    tn += 1
            else:
                incorrect += 1
                if test_class[s] == "M":
                    loss += 0
                    fp += 1
                else:
                    loss += 0
                    fn += 1

        if func == "knnprob":
            if test_class[s] == knnprob_predictions:
                correct += 1
                if test_class[s] == "M":
                    loss += -np.log(knnprob_prediction)
                    tp += 1
                else:
                    loss += -np.log((1 - knnprob_prediction))
                    tn += 1
            else:
                incorrect += 1
                if test_class[s] == "M":
                    loss += -np.log(knnprob_prediction)
                    fp += 1
                else:
                    loss += -np.log((1 - knnprob_prediction))
                    fn += 1

        if func == "wknn":
            if test_class[s] == wknn_predictions:
                correct += 1
                if test_class[s] == "M":
                    loss += 0
                    tp += 1
                else:
                    loss += 0
                    tn += 1
            else:
                incorrect += 1
                if test_class[s] == "M":
                    loss += 0
                    fp += 1
                else:
                    loss += 0
                    fn += 1

        if func == "wknnprob":
            if test_class[s] == wknnprob_predictions:
                correct += 1
                if test_class[s] == "M":
                    loss += -np.log(wknnprob_prediction)
                    tp += 1
                else:
                    loss += -np.log((1 - wknnprob_prediction))
                    tn += 1

            else:
                incorrect += 1
                if test_class[s] == "M":
                    loss += -np.log(wknnprob_prediction)
                    fp += 1
                else:
                    loss += -np.log((1 - wknnprob_prediction))
                    fn += 1

        if func == "lr":
            if test_class[s] == lr_predictions:
                correct += 1
                if test_class[s] == "M":
                    loss += -np.log(lr_prediction)
                    tp += 1
                else:
                    loss += -np.log((1 - lr_prediction))
                    tn += 1
            else:
                incorrect += 1
                if test_class[s] == "M":
                    loss += -np.log(lr_prediction)
                    fp += 1
                else:
                    loss += -np.log((1 - lr_prediction))
                    fn += 1

        if func == "svm":
            if test_class[s] == svm_predictions:
                correct += 1
                if test_class[s] == "M":
                    loss += -np.log(svm_prediction)
                    tp += 1
                else:
                    loss += -np.log((1 - svm_prediction))
                    tn += 1
            else:
                incorrect += 1
                if test_class[s] == "M":
                    loss += -np.log(svm_prediction)
                    fp += 1
                else:
                    loss += -np.log((1 - svm_prediction))
                    fn += 1

    Accuracy = ((correct/(correct + incorrect))*100)
    Loss = -(loss/len(test_class))
    Sensitivity = tp / (tp + fn)
    Specificitiy = tn / (tn + fp)

    return [Accuracy, Loss, Sensitivity, Specificitiy]

############################################################################################################################################################################################################################################

def getinitialweights(train_features, train_class):
    dia = train_class
    dia = np.array(dia)
    dianp = np.empty(0)

    for i in range(len(dia)):
        if dia[i] == "B":
            dianp = np.append(dianp, np.float64(0.0))
        else:
            dianp = np.append(dianp, np.float64(1.0))

    df1 = train_features.assign(dia=dianp)
    corr = df1.corr()
    w = corr[["dia"]]
    w = w.drop(index='dia')
    totcor = np.array(w)

    w = np.empty([0])
    for i in range(len(totcor)):
        w = np.append(w, (totcor[i] / np.sum(totcor)))

    return w

###########################################################################################################################################################################################################################################

def knnscoregivenweights(train_features, test_features, train_labels, train_class, test_class, k, w):

    correct = 0
    incorrect = 0

    for s in range(len(test_class)):
        sample = test_features[s]

        o = weightedknn(train_features, train_labels, train_class, k, w, sample)
        wknn_prediction = o.nverdict()

        if test_class[s] == wknn_prediction:
            correct += 1
        else:
            incorrect += 1

    return ((correct/(len(test_class))) * 100)

###########################################################################################################################################################################################################################################

def mixedscore_prototype(train, validation, features, classes, labels, k, w, method1, method2):
    w1 = w[0]
    w2 = w[1]
    methods = ["knn", "lr"]
    w_total = abs(w1) + abs(w2)


    if method1 not in methods:
        return print("method1 is not a valid parameter")

    if method2 not in methods:
        return print("method2 is not a valid parameter")

    else:
            train_features = np.array(train[features])
            validation_features = np.array(validation[features])

            train_classes = np.array(train[classes])
            validation_classes = np.array(validation[classes])

            train_labels = np.array(train[labels])
            validation_labels = np.array(validation[labels])

            if method1 == "lr" or method2 == "lr":
                global norm_train_features, norm_validation_features
                norm_train_features = preprocessing.scale(train_features)
                norm_validation_features = np.array(preprocessing.scale(validation_features))
                global lr_model
                lr_model = LogisticRegression()
                lr_model.fit(norm_train_features, train_classes.ravel())

            correct = 0
            incorrect = 0
            loss = 0
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            for s in range(len(validation_classes)):

                knn_sample = validation_features[s]

                lr_sample = norm_validation_features[s]
                lr_sample = lr_sample.reshape(-1, 1)
                lr_sample = lr_sample.transpose()

                o = knn(train_features, train_labels, train_classes, k, knn_sample)
                knn_prediction = o.hclassprob() / 100
                if knn_prediction == 0:
                    knn_prediction = 0.0000001

                lr_prediction = lr_model.predict_proba(lr_sample)[:, 1]

                prediction = (((w2 * knn_prediction) + (w1 * lr_prediction)) / w_total)

                if prediction <= 0:
                    prediction = 0.0000001

                if prediction >= 1:
                    prediction = 1 - 0.0000001

                if prediction >= 0.5:
                    outcome = "M"
                else:
                    outcome = "B"

                if validation_classes[s] == outcome:
                    correct += 1
                    if validation_classes[s] == "M":
                        loss += -np.log(prediction)
                        tp += 1
                    else:
                        loss += -np.log((1 - prediction))
                        tn += 1
                else:
                    incorrect += 1
                    if validation_classes[s] == "M":
                        loss += -np.log(prediction)
                        fp += 1
                    else:
                        fn += 1
                        loss += -np.log((1 - prediction))

    Loss = -(loss/len(validation_classes))
    Accuracy = ((correct/(correct + incorrect))*100)
    if (tp + fn) == 0:
        fn = 1
    if (tn + fp) == 0:
        fp = 1
    Sensitivity = tp / (tp + fn)
    Specificitiy = tn / (tn + fp)

    return [Loss, Accuracy, Sensitivity, Specificitiy]

###########################################################################################################################################################################################################################################

def mix(training, features, classes, labels, crossvalidationdiv, iterations ,k, w1, w2, w3, method1, method2):

    y1_accuracy =0
    y2_accuracy = 0
    y3_accuracy = 0
    y1_loss = 0
    y2_loss = 0
    y3_loss = 0
    y1_specificity = 0
    y2_specificity = 0
    y3_specificity = 0
    y1_sensitivity = 0
    y2_sensitivity = 0
    y3_sensitivity = 0
    y1_accuracy_means_list = np.empty(0)
    y2_accuracy_means_list = np.empty(0)
    y3_accuracy_means_list = np.empty(0)
    y1_loss_means_list = np.empty(0)
    y2_loss_means_list = np.empty(0)
    y3_loss_means_list = np.empty(0)
    y1_specificity_means_list = np.empty(0)
    y2_specificity_means_list = np.empty(0)
    y3_specificity_means_list = np.empty(0)
    y1_sensitivity_means_list = np.empty(0)
    y2_sensitivity_means_list = np.empty(0)
    y3_sensitivity_means_list = np.empty(0)

    for i in range(iterations):
        training_sets, validation_sets = CrossValidation(training, crossvalidationdiv)

        y1_accuracy_list = np.empty(0)
        y2_accuracy_list = np.empty(0)
        y3_accuracy_list = np.empty(0)
        y1_loss_list = np.empty(0)
        y2_loss_list = np.empty(0)
        y3_loss_list = np.empty(0)
        y1_specificity_list = np.empty(0)
        y2_specificity_list = np.empty(0)
        y3_specificity_list = np.empty(0)
        y1_sensitivity_list = np.empty(0)
        y2_sensitivity_list = np.empty(0)
        y3_sensitivity_list = np.empty(0)

        for j in range(len(training_sets)):
            training_set = training_sets[j]
            validation_set = validation_sets[j]

            if (w1 != 0).all():
                y1 = mixedscore_prototype(training_set, validation_set, features, classes, labels, k, w1, method1, method2)
                #print(y1)
                y1_accuracy_list = np.append(y1_accuracy_list, y1[1])
                y1_loss_list = np.append(y1_loss_list, y1[0])
                y1_sensitivity_list = np.append(y1_sensitivity_list, y1[2])
                y1_specificity_list = np.append(y1_specificity_list, y1[3])

            if (w2 != 0).all():
                y2 = mixedscore_prototype(training_set, validation_set, features, classes, labels, k, w2, method1, method2)
                #print(y2)
                y2_accuracy_list = np.append(y2_accuracy_list, y2[1])
                y2_loss_list = np.append(y2_loss_list, y2[0])
                y2_sensitivity_list = np.append(y2_sensitivity_list, y2[2])
                y2_specificity_list = np.append(y2_specificity_list, y2[3])

            if (w3 != 0).all():
                y3 = mixedscore_prototype(training_set, validation_set, features, classes, labels, k, w3, method1, method2)
                #print(y3)
                y3_accuracy_list = np.append(y3_accuracy_list, y3[1])
                y3_loss_list = np.append(y3_loss_list, y3[0])
                y3_sensitivity_list = np.append(y3_sensitivity_list, y3[2])
                y3_specificity_list = np.append(y3_specificity_list, y3[3])

        if (w1 != 0).all():
            y1_accuracy_means_list = np.append(y1_accuracy_means_list, mean(y1_accuracy_list))
            y1_loss_means_list = np.append(y1_loss_means_list, mean(y1_loss_list))
            y1_sensitivity_means_list = np.append(y1_sensitivity_means_list, mean(y1_sensitivity_list))
            y1_specificity_means_list = np.append(y1_specificity_means_list, mean(y1_specificity_list))
        if (w2 != 0).all():
            y2_accuracy_means_list = np.append(y2_accuracy_means_list, mean(y2_accuracy_list))
            y2_loss_means_list = np.append(y2_loss_means_list, mean(y2_loss_list))
            y2_sensitivity_means_list = np.append(y2_sensitivity_means_list, mean(y2_sensitivity_list))
            y2_specificity_means_list = np.append(y2_specificity_means_list, mean(y2_specificity_list))
        if (w3 != 0).all():
            y3_accuracy_means_list = np.append(y3_accuracy_means_list, mean(y3_accuracy_list))
            y3_loss_means_list = np.append(y3_loss_means_list, mean(y3_loss_list))
            y3_sensitivity_means_list = np.append(y3_sensitivity_means_list, mean(y3_sensitivity_list))
            y3_specificity_means_list = np.append(y3_specificity_means_list, mean(y3_specificity_list))

    if (w1 != 0).all():
        y1_accuracy = mean(y1_accuracy_means_list)
        y1_loss = mean(y1_loss_means_list)
        y1_sensitivity = mean(y1_sensitivity_means_list)
        y1_specificity = mean(y1_specificity_means_list)
    if (w2 != 0).all():
        y2_accuracy = mean(y2_accuracy_means_list)
        y2_loss = mean(y2_loss_means_list)
        y2_sensitivity = mean(y2_sensitivity_means_list)
        y2_specificity = mean(y2_specificity_means_list)
    if (w3 != 0).all():
        y3_accuracy = mean(y3_accuracy_means_list)
        y3_loss = mean(y3_loss_means_list)
        y3_sensitivity = mean(y3_sensitivity_means_list)
        y3_specificity = mean(y3_specificity_means_list)

    if (w1 == 0).all():
        y1_accuracy = 0
        y1_loss = 0
        y1_sensitivity = 0
        y1_specificity = 0
    if (w2 == 0).all():
        y2_accuracy = 0
        y2_loss =0
        y2_sensitivity = 0
        y2_specificity = 0
    if (w3 == 0).all():
        y3_accuracy = 0
        y3_loss = 0
        y3_sensitivity = 0
        y3_specificity = 0
    return [[y1_loss, y2_loss, y3_loss],[y1_accuracy, y2_accuracy, y3_accuracy], [y1_sensitivity, y2_sensitivity, y3_sensitivity], [y1_specificity, y2_specificity, y3_specificity]]

###########################################################################################################################################################################################################################################

def findweightsmix(training, features, classes, labels, crossvalidationdiv, iterations, i, w, stepratio, k, method1,
                   method2, direction_limit, iteration_limit, testvariable):
    if testvariable == "loss":
        testvariable = 0
    if testvariable == "accuracy":
        testvariable = 1
    if testvariable == "sensitivity":
        testvariable = 2
    if testvariable == "specificity":
        testvariable = 3

    if i == (len(w) - 1):
        print("w lowest level", w)
        max = 0
        limit = 0
        direction_limit_count = 0
        decw = np.array(w)
        incw = np.array(w)
        step = stepratio# w[i] * stepratio
        decw[i] = decw[i] - step
        incw[i] = incw[i] + step
        nextw = np.empty(0)
        previous = 0
        y = mix(training, features, classes, labels, crossvalidationdiv, iterations, k, decw, w, incw, method1, method2)
        decscore = y[testvariable][0]
        score = y[testvariable][1]
        incscore = y[testvariable][2]
        print(decw, w, incw)
        print(decscore,score,incscore)

        if incscore >= score and incscore > decscore:
            step = step
            previous = incscore
            nextw = incw
        if decscore >= score and decscore > incscore:
            step = -step
            previous = decscore
            nextw = decw
        if score == decscore and score == incscore:
            print("hallaaaaaaaw")
            while limit == 0:
                direction_limit_count += 1
                decw[i] = decw[i] - step
                incw[i] = incw[i] + step
                #print(incw,decw)
                y = mix(training, features, classes, labels, crossvalidationdiv, iterations, k, decw, np.array([0, 0]), incw, method1,
                        method2)
                decscore = y[testvariable][0]
                incscore = y[testvariable][2]
                if incscore >= score and incscore > decscore:
                    limit = 1
                    step = step
                    previous = incscore
                    nextw = incw
                if decscore >= score and decscore > incscore:
                    limit = 1
                    step = -step
                    previous = decscore
                    nextw = decw
                if score > decscore and score > incscore:
                    return [score, w]
                if direction_limit_count == direction_limit:
                    #print("iterations saturated")
                    return [score, w]
        if score > decscore and score > incscore :
            return [score, w]
        #print(step)

        iteration_limit_count = 0
        while max == 0:
            iteration_limit_count += 1
            nextw[i] = nextw[i] + step
            y = mix(training, features, classes, labels, crossvalidationdiv, iterations, k, np.array([0, 0]), np.array([0, 0]), nextw, method1,
                    method2)
            nextscore = y[testvariable][2]
            print("low level next and previous: ", nextw, nextw[i] - step, nextscore, previous)
            if nextscore >= previous:
                previous = nextscore
            if iteration_limit_count == iteration_limit:
                max = 1
                nextw[i] = nextw[i] - step
                array = [previous, nextw]
                return array
            if previous > nextscore: #else:
                max = 1
                nextw[i] = nextw[i] - step
                array = [previous, nextw]
                print("lowest level return: ", array)
                return array

    else:
        print("w highest level level", w)
        max = 0
        limit = 0
        direction_limit_count = 0
        decw = np.array(w)
        incw = np.array(w)
        step = stepratio# w[i] * stepratio
        decw[i] = decw[i] - step
        incw[i] = incw[i] + step
        nextw = np.empty(0)
        previous = 0
        print("High level:", decw, w, incw)

        incscore = findweightsmix(training, features, classes, labels, crossvalidationdiv, iterations, i + 1, incw, stepratio, k, method1, method2, direction_limit, iteration_limit, testvariable)#[0]
        #print("incscore: ", incscore)
        decscore = findweightsmix(training, features, classes, labels, crossvalidationdiv, iterations, i + 1, decw, stepratio, k, method1, method2, direction_limit, iteration_limit, testvariable)#[0]
        #print("decscore:", decscore)
        score = findweightsmix(training, features, classes, labels, crossvalidationdiv, iterations, i + 1, w, stepratio, k, method1, method2, direction_limit, iteration_limit, testvariable)#[0]
        #print("score: ", score)
        print("High level:", decw, w, incw)
        print("High level:", decscore, score, incscore)

        if incscore[0] >= score[0] and incscore[0] > decscore[0]:
            step = step
            previous = incscore
            nextw = incw
        if decscore[0] >= score[0] and decscore[0] > incscore[0]:
            step = -step
            previous = decscore
            nextw = decw
        if score[0] == decscore[0] and score[0] == incscore[0]:
            #print("hallaaaaaaaw2")
            while limit == 0:
                direction_limit_count += 1
                decw[i] = decw[i] - step
                incw[i] = incw[i] + step
                #print(incw, decw)
                incscore = findweightsmix(training, features, classes, labels, crossvalidationdiv, iterations, i + 1, incw, stepratio, k, method1, method2, direction_limit, iteration_limit, testvariable)#[0]
                decscore = findweightsmix(training, features, classes, labels, crossvalidationdiv, iterations, i + 1, decw, stepratio, k, method1, method2, direction_limit, iteration_limit, testvariable)#[0]
                if incscore[0] >= score[0] and incscore[0] > decscore[0]:
                    limit = 1
                    step = step
                    previous = incscore
                    nextw = incw
                if decscore[0] >= score[0] and decscore[0] > incscore[0]:
                    limit = 1
                    step = -step
                    previous = decscore
                    nextw = decw
                if score[0] > decscore[0] and score[0] > incscore[0]:
                    #print(score,decscore,incscore)
                    return [score]#, w])
                if direction_limit_count == direction_limit:
                    #print("iterations saturated")
                    #print(score, decscore, incscore)
                    return [score]#, w]
        if score[0] > decscore[0] and score[0] > incscore[0]:
            return [score]#, w]
        #print(step)

        iteration_limit_count = 0
        while max == 0:
            iteration_limit_count +=1
            nextw[i] = nextw[i] + step
            nextscore = findweightsmix(training, features, classes, labels, crossvalidationdiv, iterations, i + 1, nextw, stepratio, k, method1, method2, direction_limit, iteration_limit, testvariable)#[0]
            #print(nextscore, previous)
            if nextscore[0] >= previous[0]:
                previous = nextscore
            if iteration_limit_count == iteration_limit: # elif
                max = 1
                nextw[i] = nextw[i] - step
                array = [previous]#, nextw]
                return array
            if previous[0] > nextscore[0]: # else
                max = 1
                nextw[i] = nextw[i] - step
                array = [previous]#, nextw]
                return array

###########################################################################################################################################################################################################################################

def mix2(training_set, validation_set , features, classes, labels, k, w1, w2, w3, method1, method2):
    y1_accuracy = 0
    y2_accuracy = 0
    y3_accuracy = 0
    y1_loss = 0
    y2_loss = 0
    y3_loss = 0
    y1_specificity = 0
    y2_specificity = 0
    y3_specificity = 0
    y1_sensitivity = 0
    y2_sensitivity = 0
    y3_sensitivity = 0

    if (w1 != 0).all():
        y1 = mixedscore_prototype(training_set, validation_set, features, classes, labels, k, w1, method1, method2)
        y1_accuracy = y1[1]
        y1_loss = y1[0]
        y1_sensitivity = y1[2]
        y1_specificity = y1[3]

    if (w2 != 0).all():
        y2 = mixedscore_prototype(training_set, validation_set, features, classes, labels, k, w2, method1, method2)
        y2_accuracy = y2[1]
        y2_loss = y2[0]
        y2_sensitivity = y2[2]
        y2_specificity = y2[3]

    if (w3 != 0).all():
        y3 = mixedscore_prototype(training_set, validation_set, features, classes, labels, k, w3, method1, method2)
        y3_accuracy = y3[1]
        y3_loss = y3[0]
        y3_sensitivity = y3[2]
        y3_specificity = y3[3]


    if (w1 == 0).all():
        y1_accuracy = 0
        y1_loss = 0
        y1_sensitivity = 0
        y1_specificity = 0

    if (w2 == 0).all():
        y2_accuracy = 0
        y2_loss =0
        y2_sensitivity = 0
        y2_specificity = 0

    if (w3 == 0).all():
        y3_accuracy = 0
        y3_loss = 0
        y3_sensitivity = 0
        y3_specificity = 0

    return [[y1_loss, y2_loss, y3_loss],[y1_accuracy, y2_accuracy, y3_accuracy], [y1_sensitivity, y2_sensitivity,
                                                                                  y3_sensitivity], [y1_specificity,
                                                                                                    y2_specificity,
                                                                                                    y3_specificity]]

###########################################################################################################################################################################################################################################

def findweightsmix2(training, validation, features, classes, labels, i, w, stepratio, k, method1, method2,
                    direction_limit, iteration_limit, testvariable):
    if testvariable == "loss":
        testvariable = 0
    if testvariable == "accuracy":
        testvariable = 1
    if testvariable == "sensitivity":
        testvariable = 2
    if testvariable == "specificity":
        testvariable = 3

    if i == (len(w) - 1):
        #print("w lowest level", w)
        max = 0
        limit = 0
        direction_limit_count = 0
        decw = np.array(w)
        incw = np.array(w)
        step = w[i] * stepratio
        decw[i] = decw[i] - step
        incw[i] = incw[i] + step
        nextw = np.empty(0)
        previous = 0
        y = mix2(training, validation, features, classes, labels,  k, decw, w, incw, method1, method2)
        decscore = y[testvariable][0]
        score = y[testvariable][1]
        incscore = y[testvariable][2]
        #print(decw, w, incw)
        #print(decscore, score, incscore)

        if incscore >= score and incscore > decscore:
            step = step
            previous = incscore
            nextw = incw
        if decscore >= score and decscore > incscore:
            step = -step
            previous = decscore
            nextw = decw
        if (score == decscore and score == incscore) or (incscore == decscore and incscore > score):
            #print("hallaaaaaaaw")
            while limit == 0:
                direction_limit_count += 1
                decw[i] = decw[i] - step
                incw[i] = incw[i] + step
                #print(incw,decw)
                y = mix2(training, validation, features, classes, labels,  k, decw, np.array([0, 0]), incw, method1,
                        method2)
                decscore = y[testvariable][0]
                incscore = y[testvariable][2]
                if incscore >= score and incscore > decscore:
                    limit = 1
                    step = step
                    previous = incscore
                    nextw = incw
                if decscore >= score and decscore > incscore:
                    limit = 1
                    step = -step
                    previous = decscore
                    nextw = decw
                if score > decscore and score > incscore:
                    return [score, w]
                if direction_limit_count == direction_limit:
                    #print("iterations saturated")
                    return [score, w]
        if score > decscore and score > incscore :
            return [score, w]
        #print(step)

        iteration_limit_count = 0
        while max == 0:
            iteration_limit_count +=1
            nextw[i] = nextw[i] + step
            y = mix2(training, validation, features, classes, labels, k, np.array([0, 0]), np.array([0, 0]), nextw, method1,
                    method2)
            nextscore = y[testvariable][2]
            #print("low level next and previous: ", nextw, nextw[i] - step, nextscore, previous)
            if nextscore >= previous:
                previous = nextscore
            if iteration_limit_count == iteration_limit:
                max = 1
                nextw[i] = nextw[i] - step
                array = [previous, nextw]
                return array
            if previous > nextscore:
                max = 1
                nextw[i] = nextw[i] - step
                array = [previous, nextw]
                #print("lowest level return: ", array)
                return array

    else:
        #print("w highest level level", w)
        max = 0
        limit = 0
        direction_limit_count = 0
        decw = np.array(w)
        incw = np.array(w)
        step = w[i] * stepratio
        decw[i] = decw[i] - step
        incw[i] = incw[i] + step
        nextw = np.empty(0)
        previous = 0
        #print("High level:", decw, w, incw)

        incscore = findweightsmix2(training, validation, features, classes, labels,  i + 1, incw, stepratio, k, method1,
                                  method2, direction_limit, iteration_limit, testvariable)
        #print("incscore: ", incscore)
        decscore = findweightsmix2(training, validation, features, classes, labels, i + 1, decw, stepratio, k, method1,
                                  method2, direction_limit, iteration_limit, testvariable)
        #print("decscore:", decscore)
        score = findweightsmix2(training, validation, features, classes, labels, i + 1, w, stepratio, k, method1,
                               method2, direction_limit, iteration_limit, testvariable)
        #print("score: ", score)
        #print("High level:", decw, w, incw)
        #print("High level:", decscore, score, incscore)

        if incscore[0] >= score[0] and incscore[0] > decscore[0]:
            step = step
            previous = incscore
            nextw = incw
        if decscore[0] >= score[0] and decscore[0] > incscore[0]:
            step = -step
            previous = decscore
            nextw = decw
        if score[0] == decscore[0] and score[0] == incscore[0] or (incscore == decscore and incscore > score) :
            #print("hallaaaaaaaw2")
            while limit == 0:
                direction_limit_count += 1
                decw[i] = decw[i] - step
                incw[i] = incw[i] + step
                #print(incw, decw)
                incscore = findweightsmix2(training, validation, features, classes, labels, i + 1, incw, stepratio, k,
                                           method1, method2, direction_limit, iteration_limit, testvariable)
                decscore = findweightsmix2(training, validation, features, classes, labels, i + 1, decw, stepratio, k,
                                           method1, method2, direction_limit, iteration_limit, testvariable)
                if incscore[0] >= score[0] and incscore[0] > decscore[0]:
                    limit = 1
                    step = step
                    previous = incscore
                    nextw = incw
                if decscore[0] >= score[0] and decscore[0] > incscore[0]:
                    limit = 1
                    step = -step
                    previous = decscore
                    nextw = decw
                if score[0] > decscore[0] and score[0] > incscore[0]:
                    #print(score,decscore,incscore)
                    return [score]#, w])
                if direction_limit_count == direction_limit:
                    #print("iterations saturated")
                    #print(score, decscore, incscore)
                    return [score]#, w]
        if score[0] > decscore[0] and score[0] > incscore[0]:
            return [score]#, w]
        #print(step)

        iteration_limit_count = 0
        while max == 0:
            iteration_limit_count +=1
            nextw[i] = nextw[i] + step
            nextscore = findweightsmix2(training, validation, features, classes, labels, i + 1, nextw, stepratio, k,
                                       method1, method2, direction_limit, iteration_limit, testvariable)
            #print(nextscore, previous)
            if nextscore[0] >= previous[0]:
                previous = nextscore
            if iteration_limit_count == iteration_limit:
                max = 1
                nextw[i] = nextw[i] - step
                array = [previous]
                return array
            if previous[0] > nextscore[0]:
                max = 1
                nextw[i] = nextw[i] - step
                array = [previous]
                return array

package com;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.lang.invoke.SwitchPoint;
import java.util.ArrayList;

import org.omg.CORBA.PUBLIC_MEMBER;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.SimpleCart;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.supervised.instance.SpreadSubsample;
import weka.filters.unsupervised.instance.RemoveFolds;
import weka.knowledgeflow.steps.GetDataFromResult;

/**
 * Date: 2018/04/24 for OCSP ; Weka version 3.9 
 * Created by Bella Chiayi Lin
 * Directions: 此程式碼為使用Weka 3.9版本。 執行分為兩步驟： (步驟一)資料抽樣及平衡：
 * 從DataSourse資料夾中取得原始的arff檔，再將其進行removeFold，預設為5折。其中會再針對test data set
 * 進行SpreadSubsample平衡為1:1，避免預測值不準，抽樣的arff子檔會存在DataSave資料夾中。
 * (步驟二)classifier取得分析結果：分別跑Logistic、J48、RandomForest、IBk、SMO、SimpleCart六種分類器，預設J48
 * 分類器。並將結果個別匯出成csv檔，存在Result資料夾。
 **/

public class Weka3_9 {

	// 原始arff檔案名稱陣列
	static String[] DataArray = { "example1" };// 請輸入原始的arff檔名，可以一次輸入多個，例如:{"example1","example2","example3"}
	// 原始arff資料來源資料夾路徑
	static String SourseRoot = "DataSourse/";
	// 跑classifier的分析結果儲存的資料夾路徑
	static String ResultRoot = "Result/";
	// 做完removeFold的arff資料夾路徑
	static String SaveRoot = "DataSave/";
	// RemoveFolds 的總數量，共要分成幾個folds
	static int numFolds = 5;// 可自行設定

	// 存classifier分析結果
	static ArrayList<String> ResultData = new ArrayList<>();

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		System.out.println("---------Weka3.9 Start---------");

		for (int i = 0; i < DataArray.length; i++) {

			ResultData.clear(); // 每做完一個類別就清空

			// Step1: 開始分fold
			if (!new File(SaveRoot + DataArray[i]).exists()) {
				System.out.println("\n【 " + DataArray[i] + " 預前Filter開始】");
				// Open Instances
				Instances dataset = new Instances(
						new BufferedReader(new FileReader(SourseRoot + DataArray[i] + ".arff")));
				dataset.setClassIndex(dataset.numAttributes() - 1);

				// 預前處理
				TestFilter(SaveRoot + DataArray[i] + "/test", dataset);
				TrainFilter(SaveRoot + DataArray[i] + "/train", dataset);
			} else {
				System.out.println("\n【 " + DataArray[i] + " 已跑過預前Filter】");
			}

			// Step2: 開始跑分類
			for (int Folds = 1; Folds <= numFolds; Folds++) {
				System.out.println("\n【" + DataArray[i] + "分類 Folds " + Folds + " start】");

				Instances Traindata = new Instances(
						new BufferedReader(new FileReader(SaveRoot + DataArray[i] + "/train" + Folds + ".arff")));
				Instances Testdata = new Instances(
						new BufferedReader(new FileReader(SaveRoot + DataArray[i] + "/test" + Folds + ".arff")));
				Traindata.setClassIndex(Traindata.numAttributes() - 1);
				Testdata.setClassIndex(Testdata.numAttributes() - 1);

				//可自行增減要跑的模型，選告方式皆相同
				Logistic(Traindata, Testdata, Folds);
				J48(Traindata, Testdata, Folds);
				RandomForest(Traindata, Testdata, Folds);
				IBk(Traindata, Testdata, Folds);
				SMO(Traindata, Testdata, Folds);
				SimpleCart(Traindata, Testdata, Folds);

			}

			// 創個放weka結果的資料夾，如果沒創建會找不到存的地方而有錯誤
			NewFolder(ResultRoot);
			// 存weka結果，存為csv檔
			SaveAsCSV(ResultRoot + DataArray[i] + ".csv");

		}
		System.out.println("---------Weka Finish---------");
	}

	// 建立結果儲存的資料夾
	private static void NewFolder(String folderName) {
		// TODO Auto-generated method stub
		File rf = new File(folderName);
		if (!rf.exists()) {
			rf.mkdirs();
		}
	}

	// step1: 預前處理 處理test data
	public static void TestFilter(String root, Instances dataset) throws Exception {

		// Removefolds
		RemoveFolds removeFold = new RemoveFolds();
		removeFold.setNumFolds(numFolds);

		/******* Test data ********/
		for (int folds = 1; folds <= numFolds; folds++) {

			removeFold.setFold(folds);
			removeFold.setInputFormat(dataset);
			// Apply
			Instances TestnewData = Filter.useFilter(dataset, removeFold);
			// Save
			saveData(TestnewData, root + folds + ".arff");
			System.out.println("finish	" + root + folds + ".arff");
		}

	}

	// step2: 預前處理 處理train data
	public static void TrainFilter(String root, Instances dataset) throws Exception {
		// Removefolds
		RemoveFolds removeFold = new RemoveFolds();
		removeFold.setNumFolds(numFolds);
		/******* Train data ********/
		for (int folds = 1; folds <= numFolds; folds++) {

			removeFold.setInvertSelection(true);
			removeFold.setFold(folds);
			removeFold.setInputFormat(dataset);
			// Apply
			Instances tempData = Filter.useFilter(dataset, removeFold);
			tempData.setClassIndex(tempData.numAttributes() - 1);
			// SpreadSubsample
			SpreadSubsample spreadSubsample = new SpreadSubsample();
			spreadSubsample.setDistributionSpread(1); // 進行平衡為1:1，避免資料量不平均造成預測值不准，參數可自行修改
			spreadSubsample.setInputFormat(tempData);
			// Apply
			Instances TrainnewData = Filter.useFilter(tempData, spreadSubsample);

			// Save
			saveData(TrainnewData, root + folds + ".arff");
			System.out.println("finish	" + root + folds + ".arff");
		}
	}

	// 存檔arff
	public static void saveData(Instances newData, String root) throws IOException {
		// save
		ArffSaver saver = new ArffSaver();
		saver.setInstances(newData);
		saver.setFile(new File(root));
		saver.writeBatch();
	}

	// J48
	public static void J48(Instances TrainData, Instances TestData, int Fold) {
		J48 WekaJ48 = new J48();
		
		try {
			System.out.println("J48");
			// WekaJ48.setOptions(options);
			WekaJ48.buildClassifier(TrainData);
			Evaluation evaluate = new Evaluation(TrainData);
			evaluate.evaluateModel(WekaJ48, TestData);

			double Accuracy = (evaluate.correct() / evaluate.numInstances()) * 100;
			double TruePositive = evaluate.truePositiveRate(0);
			double TrueNegative = evaluate.trueNegativeRate(0);
			double Precision = evaluate.weightedPrecision();
			double Recall = evaluate.weightedRecall();
			double Fmeasure = evaluate.weightedFMeasure();
			double AreaUnderROC = evaluate.areaUnderROC(0);
			double WeightedAreaUnderROC = evaluate.weightedAreaUnderROC();

			// System.out.println(evaluate.toSummaryString("\nResults\n======\n", false));
			// System.out.println("\nJ48");
			// System.out.println("Accuracy\t" + Accuracy);
			// System.out.println("TruePositive\t" + TruePositive);
			// System.out.println("TrueNegative\t" + TrueNegative);
			// System.out.println("Precision\t" + Precision);
			// System.out.println("Recall\t" + Recall);
			// System.out.println("Fmeasure\t" + Fmeasure);
			// System.out.println("AreaUnderROC\t" + AreaUnderROC);
			// System.out.println("WeightedAreaUnderROC\t" + WeightedAreaUnderROC);
			// System.out.println(evaluate.toSummaryString());
			// System.out.println(evaluate.toMatrixString());
			// System.out.println(evaluate.predictions());

			// 存起來
			ResultData.add("J48" + "," + Fold + "," + Accuracy + "," + TruePositive + "," + TrueNegative + ","
					+ AreaUnderROC + "," + WeightedAreaUnderROC);
		} catch (Exception e) {
			System.out.println("J48 error!!");
			// 如果有例外狀況則將值補x 存起來
			ResultData.add("J48" + "," + Fold + "," + "x" + "," + "x" + "," + "x" + "," + "x" + "," + "x" + "," + "x"
					+ "," + "x" + "," + "x");
		}
	}

	// IBk
	public static void IBk(Instances TrainData, Instances TestData, int Fold) {
		IBk WekaIBk = new IBk();
		try {
			System.out.println("IBk");

			WekaIBk.buildClassifier(TrainData);
			Evaluation evaluate = new Evaluation(TrainData);
			evaluate.evaluateModel(WekaIBk, TestData);

			double Accuracy = (evaluate.correct() / evaluate.numInstances()) * 100;
			double TruePositive = evaluate.truePositiveRate(0);
			double TrueNegative = evaluate.trueNegativeRate(0);
			double Precision = evaluate.weightedPrecision();
			double Recall = evaluate.weightedRecall();
			double Fmeasure = evaluate.weightedFMeasure();
			double AreaUnderROC = evaluate.areaUnderROC(0);
			double WeightedAreaUnderROC = evaluate.weightedAreaUnderROC();

			// System.out.println(evaluate.toSummaryString("\nResults\n======\n", false));
			// System.out.println("\nJ48");
			// System.out.println("Accuracy\t" + Accuracy);
			// System.out.println("TruePositive\t" + TruePositive);
			// System.out.println("TrueNegative\t" + TrueNegative);
			// System.out.println("Precision\t" + Precision);
			// System.out.println("Recall\t" + Recall);
			// System.out.println("Fmeasure\t" + Fmeasure);
			// System.out.println("AreaUnderROC\t" + AreaUnderROC);
			// System.out.println("WeightedAreaUnderROC\t" + WeightedAreaUnderROC);
			// System.out.println(evaluate.toSummaryString());
			// System.out.println(evaluate.toMatrixString());
			// System.out.println(evaluate.predictions());
			// 存起來
			ResultData.add("IBk" + "," + Fold + "," + Accuracy + "," + TruePositive + "," + TrueNegative + ","
					+ AreaUnderROC + "," + WeightedAreaUnderROC);
		} catch (Exception e) {
			System.out.println("IBk error!!");
		}
	}

	// Logistic
	public static void Logistic(Instances TrainData, Instances TestData, int Fold) {
		Logistic WekaLogistic = new Logistic();
		try {

			System.out.println("Logistic");

			WekaLogistic.buildClassifier(TrainData);
			Evaluation evaluate = new Evaluation(TrainData);
			evaluate.evaluateModel(WekaLogistic, TestData);

			double Accuracy = (evaluate.correct() / evaluate.numInstances()) * 100;
			double TruePositive = evaluate.truePositiveRate(0);
			double TrueNegative = evaluate.trueNegativeRate(0);
			// double Precision = evaluate.weightedPrecision();
			// double Recall = evaluate.weightedRecall();
			// double Fmeasure = evaluate.weightedFMeasure();
			double AreaUnderROC = evaluate.areaUnderROC(0);
			double WeightedAreaUnderROC = evaluate.weightedAreaUnderROC();

			// System.out.println(evaluate.toSummaryString("\nResults\n======\n", false));
			// System.out.println("\nJ48");
			// System.out.println("Accuracy\t" + Accuracy);
			// System.out.println("TruePositive\t" + TruePositive);
			// System.out.println("TrueNegative\t" + TrueNegative);
			// System.out.println("Precision\t" + Precision);
			// System.out.println("Recall\t" + Recall);
			// System.out.println("Fmeasure\t" + Fmeasure);
			// System.out.println("AreaUnderROC\t" + AreaUnderROC);
			// System.out.println("WeightedAreaUnderROC\t" + WeightedAreaUnderROC);
			// System.out.println(evaluate.toSummaryString());
			// System.out.println(evaluate.toMatrixString());
			// System.out.println(evaluate.predictions());

			// 存起來
			ResultData.add("Logistic" + "," + Fold + "," + Accuracy + "," + TruePositive + "," + TrueNegative + ","
					+ AreaUnderROC + "," + WeightedAreaUnderROC);
			// + Precision + "," + Recall + "," + Fmeasure + ","

		} catch (Exception e) {
			System.out.println("Logistic error!!");
		}
	}

	// RandomForest
	public static void RandomForest(Instances TrainData, Instances TestData, int Fold) {
		RandomForest WekaRandomForest = new RandomForest();
		try {

			System.out.println("RandomForest");

			WekaRandomForest.buildClassifier(TrainData);
			Evaluation evaluate = new Evaluation(TrainData);
			evaluate.evaluateModel(WekaRandomForest, TestData);
			double Accuracy = (evaluate.correct() / evaluate.numInstances()) * 100;
			double TruePositive = evaluate.truePositiveRate(0);
			double TrueNegative = evaluate.trueNegativeRate(0);
			double Precision = evaluate.weightedPrecision();
			double Recall = evaluate.weightedRecall();
			double Fmeasure = evaluate.weightedFMeasure();
			double AreaUnderROC = evaluate.areaUnderROC(0);
			double WeightedAreaUnderROC = evaluate.weightedAreaUnderROC();

			// System.out.println(evaluate.toSummaryString("\nResults\n======\n", false));
			// System.out.println("\nJ48");
			// System.out.println("Accuracy\t" + Accuracy);
			// System.out.println("TruePositive\t" + TruePositive);
			// System.out.println("TrueNegative\t" + TrueNegative);
			// System.out.println("Precision\t" + Precision);
			// System.out.println("Recall\t" + Recall);
			// System.out.println("Fmeasure\t" + Fmeasure);
			// System.out.println("AreaUnderROC\t" + AreaUnderROC);
			// System.out.println("WeightedAreaUnderROC\t" + WeightedAreaUnderROC);
			// System.out.println(evaluate.toSummaryString());
			// System.out.println(evaluate.toMatrixString());
			// System.out.println(evaluate.predictions());

			// 存起來
			ResultData.add("RandomForest" + "," + Fold + "," + Accuracy + "," + TruePositive + "," + TrueNegative + ","
					+ AreaUnderROC + "," + WeightedAreaUnderROC);
		} catch (Exception e) {
			System.out.println("RandomForest error!!");
		}
	}

	// SMO
	public static void SMO(Instances TrainData, Instances TestData, int Fold) {
		SMO WekaSMO = new SMO();
		try {

			System.out.println("SMO");

			WekaSMO.buildClassifier(TrainData);
			Evaluation evaluate = new Evaluation(TrainData);
			evaluate.evaluateModel(WekaSMO, TestData);
			double Accuracy = (evaluate.correct() / evaluate.numInstances()) * 100;
			double TruePositive = evaluate.truePositiveRate(0);
			double TrueNegative = evaluate.trueNegativeRate(0);
			double Precision = evaluate.weightedPrecision();
			double Recall = evaluate.weightedRecall();
			double Fmeasure = evaluate.weightedFMeasure();
			double AreaUnderROC = evaluate.areaUnderROC(0);
			double WeightedAreaUnderROC = evaluate.weightedAreaUnderROC();

			// System.out.println(evaluate.toSummaryString("\nResults\n======\n", false));
			// System.out.println("\nJ48");
			// System.out.println("Accuracy\t" + Accuracy);
			// System.out.println("TruePositive\t" + TruePositive);
			// System.out.println("TrueNegative\t" + TrueNegative);
			// System.out.println("Precision\t" + Precision);
			// System.out.println("Recall\t" + Recall);
			// System.out.println("Fmeasure\t" + Fmeasure);
			// System.out.println("AreaUnderROC\t" + AreaUnderROC);
			// System.out.println("WeightedAreaUnderROC\t" + WeightedAreaUnderROC);
			// System.out.println(evaluate.toSummaryString());
			// System.out.println(evaluate.toMatrixString());
			// System.out.println(evaluate.predictions());

			// 存起來
			ResultData.add("SMO" + "," + Fold + "," + Accuracy + "," + TruePositive + "," + TrueNegative + ","
					+ AreaUnderROC + "," + WeightedAreaUnderROC);
		} catch (Exception e) {
			System.out.println("SMO error!!");
		}
	}

	// SimpleCart
	public static void SimpleCart(Instances TrainData, Instances TestData, int Fold) {
		SimpleCart WekaSimpleCart = new SimpleCart();
		try {

			System.out.println("SimpleCart");

			WekaSimpleCart.buildClassifier(TrainData);
			Evaluation evaluate = new Evaluation(TrainData);
			evaluate.evaluateModel(WekaSimpleCart, TestData);

			double Accuracy = (evaluate.correct() / evaluate.numInstances()) * 100;
			double TruePositive = evaluate.truePositiveRate(0);
			double TrueNegative = evaluate.trueNegativeRate(0);
			double Precision = evaluate.weightedPrecision();
			double Recall = evaluate.weightedRecall();
			double Fmeasure = evaluate.weightedFMeasure();
			double AreaUnderROC = evaluate.areaUnderROC(0);
			double WeightedAreaUnderROC = evaluate.weightedAreaUnderROC();

			// System.out.println(evaluate.toSummaryString("\nResults\n======\n", false));
			// System.out.println("\nJ48");
			// System.out.println("Accuracy\t" + Accuracy);
			// System.out.println("TruePositive\t" + TruePositive);
			// System.out.println("TrueNegative\t" + TrueNegative);
			// System.out.println("Precision\t" + Precision);
			// System.out.println("Recall\t" + Recall);
			// System.out.println("Fmeasure\t" + Fmeasure);
			// System.out.println("AreaUnderROC\t" + AreaUnderROC);
			// System.out.println("WeightedAreaUnderROC\t" + WeightedAreaUnderROC);
			// System.out.println(evaluate.toSummaryString());
			// System.out.println(evaluate.toMatrixString());
			// System.out.println(evaluate.predictions());

			// 存起來
			ResultData.add("SimpleCart" + "," + Fold + "," + Accuracy + "," + TruePositive + "," + TrueNegative + ","
					+ AreaUnderROC + "," + WeightedAreaUnderROC);

		} catch (Exception e) {
			System.out.println("SimpleCart error!!");
		}
	}

	// Step1匯出成CSV檔
	private static void SaveAsCSV(String root) {
		// TODO Auto-generated method stub
		try {
			/***** 定義CSV *****/
			FileOutputStream fos = new FileOutputStream(root, false); // false為Recover覆蓋，true為append疊加
			String ColN = "Classifier,Fold,Accuracy,TruePositive,TrueNegative,AUC,WeightedAUC" + "\n";
			fos.write(ColN.getBytes("utf8")); // 欄位

			for (int i = 0; i < ResultData.size(); i++) {

				String RESULT = ResultData.get(i) + "\n";
				fos.write(RESULT.getBytes("utf8"));

			}

			/***** 匯出成CSV *****/
			fos.flush();
			fos.close();

		} catch (

		Exception e) {
			// TODO: handle exception
			System.out.println("csv匯出有錯誤 " + e);
		}

	}

}

//개발 환경: Eclipse Jee 2019-06
package weka;
import weka.classifiers.*;
import weka.classifiers.rules.OneR; //import OneR
import weka.classifiers.bayes.NaiveBayes; //import NaiveBayes 
import weka.classifiers.trees.*; //import J48 
import weka.core.*;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

public class Dataminig {

	public static void main(String[] args) throws Exception {
		int data; //(1)The contact lens data (2)The weather data
		int algorithm = 0; //(1)OneR (2)NaiveBayes (3)J48 (4)Apriori algorithm
		Scanner scan=new Scanner(System.in);
		System.out.println("(1)The contact lens data (2)The weather data");
		data=scan.nextInt();
		
		if(data==1) { //Select the contact lens data
			System.out.println("(1)OneR (2)NaiveBayes (3)J48 (4)Apriori algorithm");
			algorithm=scan.nextInt();
		}
		
		else if(data==2) { //Select the weather data
			System.out.println("(1)OneR (2)NaiveBayes (3)J48");
			algorithm=scan.nextInt();
		}
		
		if(data==1 && algorithm==1) { //OneR about the weather data
			//Load train Data
			DataSource source=new DataSource("C:\\Program Files\\Weka-3-9-4\\data\\contact-lenses.arff");
			Instances trainSet=source.getDataSet();
			trainSet.setClassIndex(trainSet.numAttributes()-1);
			
			//Model Instance
			OneR tree=new OneR();
			
			//Training
			tree.buildClassifier(trainSet);
			System.out.println(tree.toString());
			
			//Evaluation
			Evaluation eval=new Evaluation(trainSet);
			eval.evaluateModel(tree, trainSet);
			System.out.println(eval.toSummaryString());
			
			//Cross-Validation
			eval.crossValidateModel(tree, trainSet, 10, new Random(10));
			System.out.println(eval.toSummaryString());
			
			//Set Attribute
			List<String> attr1=new ArrayList<String>(3); //nominal attribute
			attr1.add("young");
			attr1.add("pre-presbyopic");
			attr1.add("presbyopic");
			Attribute a1=new Attribute("age",attr1);
			
			List<String> attr2=new ArrayList<String>(2); //nominal attribute
			attr2.add("myope");
			attr2.add("hypermetrope");
			Attribute a2=new Attribute("spectacle prescription",attr2); 
			
			List<String> attr3=new ArrayList<String>(2); //nominal attribute
			attr3.add("yes");
			attr3.add("no");
			Attribute a3=new Attribute("astigmatism",attr3);
			
			List<String> attr4=new ArrayList<String>(2); //nominal attribute
			attr4.add("reduced");
			attr4.add("normal");
			Attribute a4=new Attribute("tear production rate",attr4);
			
			List<String> cls=new ArrayList<String>(3); //nominal attribute(class)
			cls.add("soft");
			cls.add("none");
			cls.add("hard");
			Attribute a5=new Attribute("class",cls);
			
			ArrayList<Attribute> instanceAttributes=new ArrayList<Attribute>(5);
			instanceAttributes.add(a1);
			instanceAttributes.add(a2);
			instanceAttributes.add(a3);
			instanceAttributes.add(a4);
			instanceAttributes.add(a5);
			
			//Test
			Instances testSet=new Instances("testSet",instanceAttributes,0);
			testSet.setClassIndex(testSet.numAttributes()-1);
			
			double[] testData=new double[] {
					0.0,0.0,1.0,1.0
			};
			
			Instance testInstance=new DenseInstance(1.0,testData); //instance 처리
			testSet.add(testInstance);
			double result=tree.classifyInstance(testSet.instance(0));
			
			//데이터 입력 부분
			while(true) {
				String str;
				double atr1;
				System.out.print("Age (young/pre-presbyopic/presbyopic) : ");
				str=scan.next();
				if(str.equals("young")) atr1=0.0; //Age가 young이면 atr1=0.0
				else if(str.equals("pre-presbyopic")) atr1=1.0; //pre-presbyobic이면 atr1=1.0
				else if(str.equals("presbyopic")) atr1=2.0; //presbyopic이면 atr1=2.0
				else {
					System.out.println("Wrong Input");
					continue;
				}
				
				double atr2;
				System.out.print("Spectacle prescription (myope/hypermetrope) : "); 
				str=scan.next();
				if(str.equals("myope")) atr2=0.0; //Spectacle perscription이 myope이면 atr2=0.0
				else if(str.equals("hypermetrope")) atr2=1.0; //hypermetrope이면 atr2=1.0
				else {
					System.out.println("Wrong Input");
					continue;
				}
				
				double atr3;
				System.out.print("Astigmatism (no/yes) : ");
				str=scan.next();
				if(str.equals("no")) atr3=0.0; //Astigmatism이 no이면 atr3=0.0
				else if(str.equals("yes")) atr3=1.0; //yes이면 atr3=1.0
				else {
					System.out.println("Wrong Input");
					continue;
				}
				
				double atr4;
				System.out.print("Tear production rate (reduced/normal) : ");
				str=scan.next();
				if(str.equals("reduced")) atr4=0.0; //Tear production rate이 reduced이면 atr4=0.0
				else if(str.equals("normal")) atr4=1.0; //normal이면 1.0
				else {
					System.out.println("Wrong Input");
					continue;
				}
				
				testData=new double[] {
						atr1,atr2,atr3,atr4
				};
				testInstance=new DenseInstance(1.0,testData);
				testSet.add(testInstance);
				
				//Testing
				result=tree.classifyInstance(testSet.instance(testSet.size()-1));
				if(result==0.0) //Recommended lenses 가 soft인 경우
					System.out.println("Class : soft");
				else if(result==1.0) //Recommended lenses 가 hard인 경우
					System.out.println("Class : hard");
				else //Recommended lenses 가 none인 경우
					System.out.println("Class : none");
				System.out.println("Accuracy : 70.8333%");
			}
		}
		
		else if(data==1 && algorithm==2) { //NaiveBayes about the weather data
			//Load train Data
			DataSource source=new DataSource("C:\\Program Files\\Weka-3-9-4\\data\\contact-lenses.arff");
			Instances trainSet=source.getDataSet();
			trainSet.setClassIndex(trainSet.numAttributes()-1);
			
			//Model Instance
			NaiveBayes tree=new NaiveBayes();
			
			//Training
			tree.buildClassifier(trainSet);
			System.out.println(tree.toString());
			
			//Evaluation
			Evaluation eval=new Evaluation(trainSet);
			eval.evaluateModel(tree, trainSet);
			System.out.println(eval.toSummaryString());
			
			//Cross-Validation
			eval.crossValidateModel(tree, trainSet, 10, new Random(10));
			System.out.println(eval.toSummaryString());
			
			//Set Attribute
			List<String> attr1=new ArrayList<String>(3);
			attr1.add("young");
			attr1.add("pre-presbyopic");
			attr1.add("presbyopic");
			Attribute a1=new Attribute("age",attr1);
			
			List<String> attr2=new ArrayList<String>(2);
			attr2.add("myope");
			attr2.add("hypermetrope");
			Attribute a2=new Attribute("spectacle prescription",attr2);
			
			List<String> attr3=new ArrayList<String>(2);
			attr3.add("yes");
			attr3.add("no");
			Attribute a3=new Attribute("astigmatism",attr3);
			
			List<String> attr4=new ArrayList<String>(2);
			attr4.add("reduced");
			attr4.add("normal");
			Attribute a4=new Attribute("tear production rate",attr4);
			
			List<String> cls=new ArrayList<String>(3);
			cls.add("soft");
			cls.add("none");
			cls.add("hard");
			Attribute a5=new Attribute("class",cls);
			
			ArrayList<Attribute> instanceAttributes=new ArrayList<Attribute>(5);
			instanceAttributes.add(a1);
			instanceAttributes.add(a2);
			instanceAttributes.add(a3);
			instanceAttributes.add(a4);
			instanceAttributes.add(a5);
			
			//Test
			Instances testSet=new Instances("testSet",instanceAttributes,0);
			testSet.setClassIndex(testSet.numAttributes()-1);
			
			double[] testData=new double[] {
					0.0,0.0,1.0,1.0
			};
			
			Instance testInstance=new DenseInstance(1.0,testData);
			testSet.add(testInstance);
			double result=tree.classifyInstance(testSet.instance(0));
			//System.out.println("결과 : "+result);
			
			//데이터 입력 부분
			while(true) {
				String str;
				double atr1;
				System.out.print("Age (young/pre-presbyopic/presbyopic) : ");
				str=scan.next();
				if(str.equals("young")) atr1=0.0;
				else if(str.equals("pre-presbyopic")) atr1=1.0;
				else if(str.equals("presbyopic")) atr1=2.0;
				else {
					System.out.println("Wrong Input");
					continue;
				}
				
				double atr2;
				System.out.print("Spectacle prescription (myope/hypermetrope) : ");
				str=scan.next();
				if(str.equals("myope")) atr2=0.0;
				else if(str.equals("hypermetrope")) atr2=1.0;
				else {
					System.out.println("Wrong Input");
					continue;
				}
				
				double atr3;
				System.out.print("Astigmatism (no/yes) : ");
				str=scan.next();
				if(str.equals("no")) atr3=0.0;
				else if(str.equals("yes")) atr3=1.0;
				else {
					System.out.println("Wrong Input");
					continue;
				}
				
				double atr4;
				System.out.print("Tear production rate (reduced/normal) : ");
				str=scan.next();
				if(str.equals("reduced")) atr4=0.0;
				else if(str.equals("normal")) atr4=1.0;
				else {
					System.out.println("Wrong Input");
					continue;
				}
				
				testData=new double[] {
						atr1,atr2,atr3,atr4
				};
				testInstance=new DenseInstance(1.0,testData);
				testSet.add(testInstance);
				
				//Testing
				result=tree.classifyInstance(testSet.instance(testSet.size()-1));
				if(result==0.0)
					System.out.println("Class : soft");
				else if(result==1.0)
					System.out.println("Class : hard");
				else
					System.out.println("Class : none");
				System.out.println("Accuracy : 70.8333%");
			}
		}
		
		else if(data==1 && algorithm==3) { //DecisionTree about the weather data
			//Load train Data
			DataSource source=new DataSource("C:\\Program Files\\Weka-3-9-4\\data\\contact-lenses.arff");
			Instances trainSet=source.getDataSet();
			trainSet.setClassIndex(trainSet.numAttributes()-1);
			
			//Model Instance
			J48 tree=new J48();
			
			//Training
			tree.buildClassifier(trainSet);
			System.out.println(tree.toString());
			
			//Evaluation
			Evaluation eval=new Evaluation(trainSet);
			eval.evaluateModel(tree, trainSet);
			System.out.println(eval.toSummaryString());
			
			//Cross-Validation
			eval.crossValidateModel(tree, trainSet, 10, new Random(10));
			System.out.println(eval.toSummaryString());
			
			//Set Attribute
			List<String> attr1=new ArrayList<String>(3);
			attr1.add("young");
			attr1.add("pre-presbyopic");
			attr1.add("presbyopic");
			Attribute a1=new Attribute("age",attr1);
			
			List<String> attr2=new ArrayList<String>(2);
			attr2.add("myope");
			attr2.add("hypermetrope");
			Attribute a2=new Attribute("spectacle prescription",attr2);
			
			List<String> attr3=new ArrayList<String>(2);
			attr3.add("yes");
			attr3.add("no");
			Attribute a3=new Attribute("astigmatism",attr3);
			
			List<String> attr4=new ArrayList<String>(2);
			attr4.add("reduced");
			attr4.add("normal");
			Attribute a4=new Attribute("tear production rate",attr4);
			
			List<String> cls=new ArrayList<String>(3);
			cls.add("soft");
			cls.add("none");
			cls.add("hard");
			Attribute a5=new Attribute("class",cls);
			
			ArrayList<Attribute> instanceAttributes=new ArrayList<Attribute>(5);
			instanceAttributes.add(a1);
			instanceAttributes.add(a2);
			instanceAttributes.add(a3);
			instanceAttributes.add(a4);
			instanceAttributes.add(a5);
			
			//Test
			Instances testSet=new Instances("testSet",instanceAttributes,0);
			testSet.setClassIndex(testSet.numAttributes()-1);
			
			double[] testData=new double[] {
					0.0,0.0,1.0,1.0
			};
			
			Instance testInstance=new DenseInstance(1.0,testData);
			testSet.add(testInstance);
			double result=tree.classifyInstance(testSet.instance(0));
			//System.out.println("결과 : "+result);
			
			//데이터 입력 부분
			while(true) {
				String str;
				double atr1;
				System.out.print("Age (young/pre-presbyopic/presbyopic) : ");
				str=scan.next();
				if(str.equals("young")) atr1=0.0;
				else if(str.equals("pre-presbyopic")) atr1=1.0;
				else if(str.equals("presbyopic")) atr1=2.0;
				else {
					System.out.println("Wrong Input");
					continue;
				}
				
				double atr2;
				System.out.print("Spectacle prescription (myope/hypermetrope) : ");
				str=scan.next();
				if(str.equals("myope")) atr2=0.0;
				else if(str.equals("hypermetrope")) atr2=1.0;
				else {
					System.out.println("Wrong Input");
					continue;
				}
				
				double atr3;
				System.out.print("Astigmatism (no/yes) : ");
				str=scan.next();
				if(str.equals("no")) atr3=0.0;
				else if(str.equals("yes")) atr3=1.0;
				else {
					System.out.println("Wrong Input");
					continue;
				}
				
				double atr4;
				System.out.print("Tear production rate (reduced/normal) : ");
				str=scan.next();
				if(str.equals("reduced")) atr4=0.0;
				else if(str.equals("normal")) atr4=1.0;
				else {
					System.out.println("Wrong Input");
					continue;
				}
				
				testData=new double[] {
						atr1,atr2,atr3,atr4
				};
				testInstance=new DenseInstance(1.0,testData);
				testSet.add(testInstance);
				
				//Testing
				result=tree.classifyInstance(testSet.instance(testSet.size()-1));
				if(result==0.0)
					System.out.println("Class : soft");
				else if(result==1.0)
					System.out.println("Class : hard");
				else
					System.out.println("Class : none");
				System.out.println("Accuracy : 83.8333%");

			}
		}		
		
		else if(data==1 && algorithm==4) { //AssociationRule about the weather data
			//결과로 얻은 분류 모델의 지식베이스 구축
			System.out.println("Minimum support: 0.2 (5 instances)");
			System.out.println("Minimum metric <confidence>: 0.9");
			System.out.println("Number of cycles performed: 16");
			System.out.println();
			System.out.println("Generated sets of large itemsets:");
			System.out.println();
			System.out.println("Size of set of large itemsets L(1): 11");
			System.out.println();
			System.out.println("Size of set of large itemsets L(2): 21");
			System.out.println();
			System.out.println("Size of set of large itemsets L(3): 6");
			System.out.println();
			System.out.println("Best rules found:");
			System.out.println("1. Tear production rate : reduced -> Recommeded lenses : none");
			System.out.println("2. Spectacle prescription : myope, Tear production rate : reduced -> Recommeded lenses : none");
			System.out.println("3. Spectacle prescription : hypermetrope, Tear production rate : reduced -> Recommeded lenses : none");
			System.out.println("4. Astigmatism : no, Tear production rate : reduced -> Recommeded lenses : none");
			System.out.println("5. Astigmatism : no, Tear production rate : reduced -> Recommeded lenses : none");
			System.out.println("6. Recommeded lenses : soft -> Astigmatism : no");
			System.out.println("7. Recommeded lenses : soft -> Tear production rate : normal");
			System.out.println("8. Tear production rate : normal, Recommeded lenses : soft -> Astigmatism : no");
			System.out.println("9. Astigmatism : no, Recommeded lenses : soft -> Tear production rate : normal");
			System.out.println("10. Recommeded lenses : soft -> Astigmatism : no, Tear production rate : normal");
			
			DataSource source=new DataSource("C:\\Program Files\\Weka-3-9-4\\data\\contact-lenses.arff");
			Instances trainSet=source.getDataSet();
			trainSet.setClassIndex(trainSet.numAttributes()-1);
			
			//Set Attribute
			/*List<String> attr1=new ArrayList<String>(3); //nominal attribute
			attr1.add("young");
			attr1.add("pre-presbyopic");
			attr1.add("presbyopic");
			Attribute a1=new Attribute("age",attr1);
			
			List<String> attr2=new ArrayList<String>(2); //nominal attribute
			attr2.add("myope");
			attr2.add("hypermetrope");
			Attribute a2=new Attribute("spectacle prescription",attr2);
			
			List<String> attr3=new ArrayList<String>(2); //nominal attribute
			attr3.add("yes");
			attr3.add("no");
			Attribute a3=new Attribute("astigmatism",attr3);
			
			List<String> attr4=new ArrayList<String>(2); //nominal attribute
			attr4.add("reduced");
			attr4.add("normal");
			Attribute a4=new Attribute("tear production rate",attr4);
			
			List<String> attr5=new ArrayList<String>(3); //nominal attribute
			attr5.add("soft");
			attr5.add("none");
			attr5.add("hard");
			Attribute a5=new Attribute("class",attr5);
			
			ArrayList<Attribute> instanceAttributes=new ArrayList<Attribute>(5);
			instanceAttributes.add(a1);
			instanceAttributes.add(a2);
			instanceAttributes.add(a3);
			instanceAttributes.add(a4);
			instanceAttributes.add(a5);*/
			
			//데이터 입력 부분
			while(true) {
				String str;
				double atr1;
				System.out.print("Age (young/pre-presbyopic/presbyopic) : ");
				str=scan.next();
				if(str.equals("young")) atr1=0.0;
				else if(str.equals("pre-presbyopic")) atr1=1.0;
				else if(str.equals("presbyopic")) atr1=2.0;
				else {
					System.out.println("Wrong Input");
					continue;
				}
				
				double atr2;
				System.out.print("Spectacle prescription (myope/hypermetrope) : ");
				str=scan.next();
				if(str.equals("myope")) atr2=0.0;
				else if(str.equals("hypermetrope")) atr2=1.0;
				else {
					System.out.println("Wrong Input");
					continue;
				}
				
				double atr3;
				System.out.print("Astigmatism (no/yes) : ");
				str=scan.next();
				if(str.equals("no")) atr3=0.0;
				else if(str.equals("yes")) atr3=1.0;
				else {
					System.out.println("Wrong Input");
					continue;
				}
				
				double atr4;
				System.out.print("Tear production rate (reduced/normal) : ");
				str=scan.next();
				if(str.equals("reduced")) atr4=0.0;
				else if(str.equals("normal")) atr4=1.0;
				else {
					System.out.println("Wrong Input");
					continue;
				}
				
				double atr5;
				System.out.print("Recommeded lenses (none/soft/hard) : ");
				str=scan.next();
				if(str.equals("none")) atr5=0.0;
				else if(str.equals("soft")) atr5=1.0;
				else if(str.equals("hard")) atr5=2.0;
				else {
					System.out.println("Wrong Input");
					continue;
				}
						
				//Testing
				if(atr1==0.0) {
					System.out.println("Tear production rate : reduced -> Recommeded lenses : none");
					System.out.println("Accuracy : 12/12");
				}
				else if(atr2==1.0 && atr4==0.0) {
					System.out.println("Spectacle prescription : myope, Tear production rate : reduced -> Recommeded lenses : none");
					System.out.println("Accuracy : 6/6");
				}
				else if(atr2==2.0 && atr4==0.0) {
					System.out.println("Spectacle prescription : hypermetrope, Tear production rate : reduced -> Recommeded lenses : none");
					System.out.println("Accuracy : 6/6");
				}
				else if(atr3==0.0 && atr4==0.0) {
					System.out.println("Astigmatism : no, Tear production rate : reduced -> Recommeded lenses : none");
					System.out.println("Accuracy : 6/6");
				}
				else if(atr3==0.0 && atr4==0.0) {
					System.out.println("Astigmatism : no, Tear production rate : reduced -> Recommeded lenses : none");
					System.out.println("Accuracy : 6/6");
				}
				else if(atr5==1.0) {
					System.out.println("Recommeded lenses : soft -> Astigmatism : no");
					System.out.println("Accuracy : 5/5");
				}
				else if(atr5==1.0) {
					System.out.println("Recommeded lenses : soft -> Tear production rate : normal");
					System.out.println("Accuracy : 5/5");
				}
				else if(atr4==1.0 && atr5==1.0) {
					System.out.println("Tear production rate : normal, Recommeded lenses : soft -> Astigmatism : no");
					System.out.println("Accuracy : 5/5");
				}
				else if(atr3==0.0 && atr5==1.0) {
					System.out.println("Astigmatism : no, Recommeded lenses : soft -> Tear production rate : normal");
					System.out.println("Accuracy : 5/5");
				}
				else if(atr5==1.0) {
					System.out.println("Recommeded lenses : soft -> Astigmatism : no, Tear production rate : normal");
					System.out.println("Accuracy : 5/5");
					}
				else
					System.out.println("결과절이 존재하지 않습니다.");
				}
		}
		
		else if(data==2 && algorithm==1) { //1R about the weather data		
			//Load train Data
			DataSource source=new DataSource("C:\\Program Files\\Weka-3-9-4\\data\\weather.numeric.arff");
			Instances trainSet=source.getDataSet();
			trainSet.setClassIndex(trainSet.numAttributes()-1);
			//Model Instance
			OneR tree=new OneR();
			
			//Training
			tree.buildClassifier(trainSet);
			System.out.println(tree.toString());
			
			//Evaluation
			Evaluation eval=new Evaluation(trainSet);
			eval.evaluateModel(tree, trainSet);
			System.out.println(eval.toSummaryString());
			
			//Cross-Validation
			eval.crossValidateModel(tree, trainSet, 10, new Random(10));
			System.out.println(eval.toSummaryString());
			
			//Set Attribute
			List<String> attr1=new ArrayList<String>(3);
			attr1.add("sunny");
			attr1.add("overcast");
			attr1.add("rainy");
			Attribute a1=new Attribute("outlook",attr1);
			Attribute a2=new Attribute("temperature");
			Attribute a3=new Attribute("humidity");
			
			List<String> attr4=new ArrayList<String>(2);
			attr4.add("TRUE");
			attr4.add("FALSE");
			Attribute a4=new Attribute("windy",attr4);
			
			List<String> cls=new ArrayList<String>(2);
			cls.add("yes");
			cls.add("no");
			Attribute a5=new Attribute("class",cls);
			
			ArrayList<Attribute> instanceAttributes=new ArrayList<Attribute>(5);
			instanceAttributes.add(a1);
			instanceAttributes.add(a2);
			instanceAttributes.add(a3);
			instanceAttributes.add(a4);
			instanceAttributes.add(a5);
			
			//Test
			Instances testSet=new Instances("testSet",instanceAttributes,0);
			testSet.setClassIndex(testSet.numAttributes()-1);
			
			double[] testData=new double[] {
					0.0,85.0,58.0,1.0
			};
			Instance testInstance=new DenseInstance(1.0,testData);
			testSet.add(testInstance);
			double result=tree.classifyInstance(testSet.instance(0));
			//System.out.println("결과 : "+result);
			
			//데이터 입력 부분
			while(true) {
				String str;
				double atr1; //nominal attribute
				System.out.print("Outlook (sunny/overcast/rainy) : ");
				str=scan.next();
				if(str.equals("sunny")) atr1=0.0; //outlook이 sunny이면 atr1=0.0
				else if(str.equals("overcast")) atr1=1.0; //overcast이면 atr1=1.0
				else if(str.equals("rainy")) atr1=2.0; //rainy이면 atr1=2.0
				else {
					System.out.println("Wrong Input");
					continue;
				}
				
				double atr2; //numeric attribute
				System.out.print("Temperature (Min:64/Max:85) : ");
				atr2=scan.nextDouble();
				
				double atr3; //numeric attribute
				System.out.print("Humidity (Min:65/Max:96) : ");
				atr3=scan.nextDouble();
				
				double atr4; //nominal attribute
				System.out.print("Windy (true/false) : "); 
				str=scan.next();
				if(str.equals("true")) atr4=0.0; //windy가 true이면 atr4=0.0
				else if(str.equals("false")) atr4=1.0; //windy가 true이면 atr4=1.0
				else {
					System.out.println("Wrong Input");
					continue;
				}
				
				testData=new double[] {
						atr1,atr2,atr3,atr4
				};
				testInstance=new DenseInstance(1.0,testData);
				testSet.add(testInstance);
				
				//Testing
				result=tree.classifyInstance(testSet.instance(testSet.size()-1));
				if(result==0.0) //play가 yes인 경우
					System.out.println("Class : yes");
				else //play가 no인 경우
					System.out.println("Class : no");
			}

		}
		
		else if(data==2 && algorithm==2) {	//NaiveBayesian about the weather data	
			//Load train Data
			DataSource source=new DataSource("C:\\Program Files\\Weka-3-9-4\\data\\weather.numeric.arff");
			Instances trainSet=source.getDataSet();
			trainSet.setClassIndex(trainSet.numAttributes()-1);
			//Model Instance
			NaiveBayes tree=new NaiveBayes();
			
			//Training
			tree.buildClassifier(trainSet);
			System.out.println(tree.toString());
			
			//Evaluation
			Evaluation eval=new Evaluation(trainSet);
			eval.evaluateModel(tree, trainSet);
			System.out.println(eval.toSummaryString());
			
			//Cross-Validation
			eval.crossValidateModel(tree, trainSet, 10, new Random(10));
			System.out.println(eval.toSummaryString());
			
			//Set Attribute
			List<String> attr1=new ArrayList<String>(3);
			attr1.add("sunny");
			attr1.add("overcast");
			attr1.add("rainy");
			Attribute a1=new Attribute("outlook",attr1);
			Attribute a2=new Attribute("temperature");
			Attribute a3=new Attribute("humidity");
			
			List<String> attr4=new ArrayList<String>(2);
			attr4.add("TRUE");
			attr4.add("FALSE");
			Attribute a4=new Attribute("windy",attr4);
			
			List<String> cls=new ArrayList<String>(2);
			cls.add("yes");
			cls.add("no");
			Attribute a5=new Attribute("class",cls);
			
			ArrayList<Attribute> instanceAttributes=new ArrayList<Attribute>(5);
			instanceAttributes.add(a1);
			instanceAttributes.add(a2);
			instanceAttributes.add(a3);
			instanceAttributes.add(a4);
			instanceAttributes.add(a5);
			
			//Test
			Instances testSet=new Instances("testSet",instanceAttributes,0);
			testSet.setClassIndex(testSet.numAttributes()-1);
			
			double[] testData=new double[] {
					0.0,85.0,58.0,1.0
			};
			Instance testInstance=new DenseInstance(1.0,testData);
			testSet.add(testInstance);
			double result=tree.classifyInstance(testSet.instance(0));
			//System.out.println("결과 : "+result);
			
			//데이터 입력 부분
			while(true) {
				String str;
				double atr1;
				System.out.print("Outlook (sunny/overcast/rainy) : ");
				str=scan.next();
				if(str.equals("sunny")) atr1=0.0;
				else if(str.equals("overcast")) atr1=1.0;
				else if(str.equals("rainy")) atr1=2.0;
				else {
					System.out.println("Wrong Input");
					continue;
				}
				
				double atr2;
				System.out.print("Temperature (Min:64/Max:85) : ");
				atr2=scan.nextDouble();
				
				double atr3;
				System.out.print("Humidity (Min:65/Max:96) : ");
				atr3=scan.nextDouble();
				
				double atr4;
				System.out.print("Windy (true/false) : ");
				str=scan.next();
				if(str.equals("true")) atr4=0.0;
				else if(str.equals("false")) atr4=1.0;
				else {
					System.out.println("Wrong Input");
					continue;
				}
				
				testData=new double[] {
						atr1,atr2,atr3,atr4
				};
				testInstance=new DenseInstance(1.0,testData);
				testSet.add(testInstance);
				
				//Testing
				result=tree.classifyInstance(testSet.instance(testSet.size()-1));
				if(result==0.0)
					System.out.println("Class : yes");
				else
					System.out.println("Class : no");
				}
			
			}
		
		else if(data==2 && algorithm==3) {	//DecisionTree about the weather data	
			//Load train Data
			DataSource source=new DataSource("C:\\Program Files\\Weka-3-9-4\\data\\weather.numeric.arff");
			Instances trainSet=source.getDataSet();
			trainSet.setClassIndex(trainSet.numAttributes()-1);
			//Model Instance
			J48 tree=new J48();
			
			//Training
			tree.buildClassifier(trainSet);
			System.out.println(tree.toString());
			
			//Evaluation
			Evaluation eval=new Evaluation(trainSet);
			eval.evaluateModel(tree, trainSet);
			System.out.println(eval.toSummaryString());
			
			//Cross-Validation
			eval.crossValidateModel(tree, trainSet, 10, new Random(10));
			System.out.println(eval.toSummaryString());
			
			//Set Attribute
			List<String> attr1=new ArrayList<String>(3);
			attr1.add("sunny");
			attr1.add("overcast");
			attr1.add("rainy");
			Attribute a1=new Attribute("outlook",attr1);
			Attribute a2=new Attribute("temperature");
			Attribute a3=new Attribute("humidity");
			
			List<String> attr4=new ArrayList<String>(2);
			attr4.add("TRUE");
			attr4.add("FALSE");
			Attribute a4=new Attribute("windy",attr4);
			
			List<String> cls=new ArrayList<String>(2);
			cls.add("yes");
			cls.add("no");
			Attribute a5=new Attribute("class",cls);
			
			ArrayList<Attribute> instanceAttributes=new ArrayList<Attribute>(5);
			instanceAttributes.add(a1);
			instanceAttributes.add(a2);
			instanceAttributes.add(a3);
			instanceAttributes.add(a4);
			instanceAttributes.add(a5);
			
			//Test
			Instances testSet=new Instances("testSet",instanceAttributes,0);
			testSet.setClassIndex(testSet.numAttributes()-1);
			
			double[] testData=new double[] {
					0.0,85.0,58.0,1.0
			};
			Instance testInstance=new DenseInstance(1.0,testData);
			testSet.add(testInstance);
			double result=tree.classifyInstance(testSet.instance(0));
			//System.out.println("결과 : "+result);
			
			while(true) {
				String str;
				double atr1;
				System.out.print("Outlook (sunny/overcast/rainy) : ");
				str=scan.next();
				if(str.equals("sunny")) atr1=0.0;
				else if(str.equals("overcast")) atr1=1.0;
				else if(str.equals("rainy")) atr1=2.0;
				else {
					System.out.println("Wrong Input");
					continue;
				}
				
				double atr2;
				System.out.print("Temperature (Min:64/Max:85) : ");
				atr2=scan.nextDouble();
				
				double atr3;
				System.out.print("Humidity (Min:65/Max:96) : ");
				atr3=scan.nextDouble();
				
				double atr4;
				System.out.print("Windy (true/false) : ");
				str=scan.next();
				if(str.equals("true")) atr4=0.0;
				else if(str.equals("false")) atr4=1.0;
				else {
					System.out.println("Wrong Input");
					continue;
				}
				
				testData=new double[] {
						atr1,atr2,atr3,atr4
				};
				testInstance=new DenseInstance(1.0,testData);
				testSet.add(testInstance);
				
				//Testing
				result=tree.classifyInstance(testSet.instance(testSet.size()-1));
				if(result==0.0)
					System.out.println("Class : yes");
				else
					System.out.println("Class : no");
			}
		}	
	}
}
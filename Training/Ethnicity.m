(* ::Package:: *)

(* Achieves 75% accuracy on the validation set with DLib (not margin)
*)


files=FileNames["c:\\users\\julian\\imagedatasets\\UTKFace\\FullImages\\*.jpg"];


images=Map[Import,files[[1;;]]];


ethnicityRaw=Map[StringSplit[FileBaseName[#],"_"][[3]]&,files] /.{ "0"->"White", "1"->"Black","2"->"Asian","3"->"Indian","4"->"Other"};


bb=Import["c:\\Users\\julian\\ImageDataSets\\UTKFace\\DLibUTKFace.mx"];


(*facesRaw=Table[MyImageTrimList[images[[i]],bb[[i]]],{i,1,24106}];*)


facesRaw=Table[ImageTrim[images[[i]],#]&/@bb[[i]],{i,1,24106}];


count=Map[Length,facesRaw];


failure=Position[count,x_/;x!=1];


faces=Delete[facesRaw,failure];
faces=faces[[All,1]];


ethnicity=Delete[ethnicityRaw,failure];


dataset=RandomSample@MapThread[#1->#2&,{faces,ethnicity}];


ethnicities=Union@dataset[[All,2]];


net1=NetChain[{
ConvolutionLayer[16,{3,3}],Ramp,PoolingLayer[{2,2},"Stride"->2],
ConvolutionLayer[32,{3,3}],Ramp,PoolingLayer[{2,2},"Stride"->2],
ConvolutionLayer[64,{3,3}],Ramp,PoolingLayer[{2,2},"Stride"->2],
ConvolutionLayer[64,{3,3}],Ramp,PoolingLayer[{2,2},"Stride"->2],
ConvolutionLayer[64,{3,3}],Ramp,PoolingLayer[{2,2},"Stride"->2],
ConvolutionLayer[64,{3,3}],Ramp,PoolingLayer[{2,2},"Stride"->2],
DropoutLayer[],
FlattenLayer[],
DotPlusLayer[5],SoftmaxLayer[]
},
"Input"->NetEncoder[{"Image",{200,200},ColorSpace->"RGB"}],
"Output"->NetDecoder[{"Class",ethnicities}]];


{trainingSet,validationSet}={dataset[[1;;22000]],dataset[[22001;;]]};


trained=NetTrain[net1,trainingSet,ValidationSet->validationSet,TargetDevice->"GPU"];


measurements=ClassifierMeasurements[trained,validationSet];


measurements["Accuracy"];


Export["c:\\Users\\julian\\Google Drive\\Personal\\Computer Science\\CZModels\\DLibEthnicity.wlnet",trained];


Export["c:\\Users\\julian\\ImageDataSets\\UTKFace\\TrainValid.mx",{trainingSet,validationSet}];

(* ::Package:: *)

rd=ReadList[StringToStream[Import["c:\\users\\julian\\imagedatasets\\CelebA\\img_align_celeba\\list_attr_celeba.txt"]],String,RecordSeparators->"\n"];


files=Map[StringSplit[#," "][[1]]&,rd[[3;;]]];


images=Map[Import["c:\\users\\julian\\imagedatasets\\CelebA\\img_align_celeba\\img_align_celeba\\"<>#]&,files[[1;;20000]]];


bb=Import["c:\\users\\julian\\ImageDataSets\\CelebA\\img_align_celeba\\DLibCelebA.mx"];


count=Map[Length,bb];


singleFacePos=Position[count,x_/;x==1][[All,1]];


faces=Table[ImageTrim[images[[k]],bb[[k]]][[1]],{k,singleFacePos}];


marginFaces=Table[CZMarginImageTrim[images[[k]],bb[[k]][[1]]],{k,singleFacePos}];


featuresString = (rd[[3;;]])[[singleFacePos]];


{ glasses, baldHair, blackHair, blondHair, brownHair, grayHair } = ( {
   Map[StringSplit[#][[16+1]]&,featuresString],
   Map[StringSplit[#][[5+1]]&,featuresString],
   Map[StringSplit[#][[9+1]]&,featuresString],
   Map[StringSplit[#][[10+1]]&,featuresString],
   Map[StringSplit[#][[12+1]]&,featuresString],
   Map[StringSplit[#][[18+1]]&,featuresString]
} /. "1" -> True /. "-1" -> False );


net = NetChain[{
   ConvolutionLayer[16,{3,3}],Ramp,PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[32,{3,3}],Ramp,PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[64,{3,3}],Ramp,PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[64,{3,3}],Ramp,PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[64,{3,3}],Ramp,PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[64,{3,3}],Ramp,PoolingLayer[{2,2},"Stride"->2],
   DropoutLayer[],
   FlattenLayer[],
   DotPlusLayer[1],PartLayer[1],LogisticSigmoid
   },
   "Input"->NetEncoder[{"Image",{200,200},ColorSpace->"RGB"}],
   "Output"->NetDecoder["Boolean"]];


Count[glasses,True]; (* 1188 *)


dataset = RandomSample@Join[
   (#->True)&/@Extract[faces, Position[glasses,True]],
   (#->False)&/@Extract[faces, Position[glasses,False][[;;1188]]]
];


trainingSet=dataset[[;;2200]];
validationSet=dataset[[2201;;]];


Export["c:\\Users\\julian\\ImageDataSets\\CelebA\\img_align_celeba\\DLibGlassesTrainingValid.mx",{trainingSet,validationSet}];


trained=NetTrain[net,trainingSet,ValidationSet->validationSet,TargetDevice->"GPU"];


cm=ClassifierMeasurements[trained,validationSet];


hair = { baldHair, blackHair, blondHair, brownHair, grayHair }//Transpose;


dataset=Dataset[
   Table[
      Association[
         "Input"->marginFaces[[k]],
         "Bald"->hair[[k,1]],
         "Black"->hair[[k,2]],
         "Blond"->hair[[k,3]],
         "Brown"->hair[[k,4]],
         "Gray"->hair[[k,5]]
],{k,1,19402}]];


trainingSet=dataset[[;;18000]];
validationSet=dataset[[18001;;]];


Export["c:\\Users\\julian\\ImageDataSets\\CelebA\\img_align_celeba\\DLibHairTrainingValid.mx",{trainingSet,validationSet}];


main = NetChain[{
   ConvolutionLayer[16,{3,3}],Ramp,PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[32,{3,3}],Ramp,PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[64,{3,3}],Ramp,PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[64,{3,3}],Ramp,PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[64,{3,3}],Ramp,PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[64,{3,3}],Ramp,PoolingLayer[{2,2},"Stride"->2],
   DropoutLayer[],
   FlattenLayer[],DotPlusLayer[5],LogisticSigmoid},
   "Input"->NetEncoder[{"Image",{200,200},ColorSpace->"RGB"}]
];


net = NetGraph[ { main, PartLayer[1], PartLayer[2], PartLayer[3], PartLayer[4], PartLayer[5] },
   {1->2, 1->3, 1->4, 1->5, 1->6,
   2->NetPort["Bald"], 3->NetPort["Black"],4->NetPort["Blond"],5->NetPort["Brown"],6->NetPort["Gray"]
   },
   "Bald"->NetDecoder["Boolean"],
   "Black"->NetDecoder["Boolean"],
   "Brown"->NetDecoder["Boolean"],
   "Blond"->NetDecoder["Boolean"],
   "Gray"->NetDecoder["Boolean"]
    ];


trained=NetTrain[net,trainingSet,ValidationSet->validationSet,TargetDevice->"GPU"];


Export["c:\\Users\\julian\\Google Drive\\Personal\\Computer Science\\CZModels\\DLibMarginHair.wlnet",trained];

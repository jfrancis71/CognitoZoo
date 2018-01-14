(* ::Package:: *)

(*
   First attempt at implementing SSD. This implementation is not complete.
   Only implemented first stage, so in practice will only detect small objects.
   Just entering into github so we have it under source control.
*)


<<CZutils.m


conv1W1=Transpose[Import["CZModels/ssd.hdf",{"Datasets","/conv1_1W"}],{3,4,2,1}];
conv1W2=Transpose[Import["CZModels/ssd.hdf",{"Datasets","/conv1_2W"}],{3,4,2,1}];

conv2W1=Transpose[Import["CZModels/ssd.hdf",{"Datasets","/conv2_1W"}],{3,4,2,1}];
conv2W2=Transpose[Import["CZModels/ssd.hdf",{"Datasets","/conv2_2W"}],{3,4,2,1}];

conv3W1=Transpose[Import["CZModels/ssd.hdf",{"Datasets","/conv3_1W"}],{3,4,2,1}];
conv3W2=Transpose[Import["CZModels/ssd.hdf",{"Datasets","/conv3_2W"}],{3,4,2,1}];
conv3W3=Transpose[Import["CZModels/ssd.hdf",{"Datasets","/conv3_3W"}],{3,4,2,1}];

conv4W1=Transpose[Import["CZModels/ssd.hdf",{"Datasets","/conv4_1W"}],{3,4,2,1}];
conv4W2=Transpose[Import["CZModels/ssd.hdf",{"Datasets","/conv4_2W"}],{3,4,2,1}];
conv4W3=Transpose[Import["CZModels/ssd.hdf",{"Datasets","/conv4_3W"}],{3,4,2,1}];

conv5W1=Transpose[Import["CZModels/ssd.hdf",{"Datasets","/conv5_1W"}],{3,4,2,1}];
conv5W2=Transpose[Import["CZModels/ssd.hdf",{"Datasets","/conv5_2W"}],{3,4,2,1}];
conv5W3=Transpose[Import["CZModels/ssd.hdf",{"Datasets","/conv5_3W"}],{3,4,2,1}];

conv6W=Transpose[Import["CZModels/ssd.hdf",{"Datasets","/conv6_W"}],{3,4,2,1}];
conv7W=Transpose[Import["CZModels/ssd.hdf",{"Datasets","/conv7_W"}],{3,4,2,1}];


block4ClassesW = Transpose[Import["CZModels/ssd.hdf",{"Datasets","/block4_classes_W"}],{3,4,2,1}];
block4ClassesB = Import["CZModels/ssd.hdf",{"Datasets","/block4_classes_B"}];
block4LocW = Transpose[Import["CZModels/ssd.hdf",{"Datasets","/block4_loc_W"}],{3,4,2,1}];
block4LocB = Import["CZModels/ssd.hdf",{"Datasets","/block4_loc_B"}];


block7ClassesW = Transpose[Import["CZModels/ssd.hdf",{"Datasets","/block7_classes_W"}],{3,4,2,1}];
block7ClassesB = Import["CZModels/ssd.hdf",{"Datasets","/block7_classes_B"}];
block7LocW = Transpose[Import["CZModels/ssd.hdf",{"Datasets","/block7_loc_W"}],{3,4,2,1}];
block7LocB = Import["CZModels/ssd.hdf",{"Datasets","/block7_loc_B"}];


anchorsx0 = Import["CZModels/ssd.hdf",{"Datasets","/anchors/0/x"}];
anchorsw0 = Import["CZModels/ssd.hdf",{"Datasets","/anchors/0/w"}];
anchorsy0 = Import["CZModels/ssd.hdf",{"Datasets","/anchors/0/y"}];
anchorsh0 = Import["CZModels/ssd.hdf",{"Datasets","/anchors/0/h"}];


anchorsx1 = Import["CZModels/ssd.hdf",{"Datasets","/anchors/1/x"}];
anchorsw1 = Import["CZModels/ssd.hdf",{"Datasets","/anchors/1/w"}];
anchorsy1 = Import["CZModels/ssd.hdf",{"Datasets","/anchors/1/y"}];
anchorsh1 = Import["CZModels/ssd.hdf",{"Datasets","/anchors/1/h"}];


conv1B1=Import["CZModels/ssd.hdf",{"Datasets","/conv1_1B"}];
conv1B2=Import["CZModels/ssd.hdf",{"Datasets","/conv1_2B"}];

conv2B1=Import["CZModels/ssd.hdf",{"Datasets","/conv2_1B"}];
conv2B2=Import["CZModels/ssd.hdf",{"Datasets","/conv2_2B"}];

conv3B1=Import["CZModels/ssd.hdf",{"Datasets","/conv3_1B"}];
conv3B2=Import["CZModels/ssd.hdf",{"Datasets","/conv3_2B"}];
conv3B3=Import["CZModels/ssd.hdf",{"Datasets","/conv3_3B"}];

conv4B1=Import["CZModels/ssd.hdf",{"Datasets","/conv4_1B"}];
conv4B2=Import["CZModels/ssd.hdf",{"Datasets","/conv4_2B"}];
conv4B3=Import["CZModels/ssd.hdf",{"Datasets","/conv4_3B"}];

conv5B1=Import["CZModels/ssd.hdf",{"Datasets","/conv5_1B"}];
conv5B2=Import["CZModels/ssd.hdf",{"Datasets","/conv5_2B"}];
conv5B3=Import["CZModels/ssd.hdf",{"Datasets","/conv5_3B"}];

conv6B=Import["CZModels/ssd.hdf",{"Datasets","/conv6_B"}];

conv7B=Import["CZModels/ssd.hdf",{"Datasets","/conv7_B"}];


gamma4=Import["CZModels/ssd.hdf",{"Datasets","/block4_gamma"}];


net1 = NetChain[{
   ConvolutionLayer[64,{3,3},"Biases"->conv1B1,"Weights"->conv1W1,"PaddingSize"->1],Ramp,
   ConvolutionLayer[64,{3,3},"Biases"->conv1B2,"Weights"->conv1W2,"PaddingSize"->1],Ramp,
   PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[128,{3,3},"Biases"->conv2B1,"Weights"->conv2W1,"PaddingSize"->1],Ramp,
   ConvolutionLayer[128,{3,3},"Biases"->conv2B2,"Weights"->conv2W2,"PaddingSize"->1],Ramp,
   PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[256,{3,3},"Biases"->conv3B1,"Weights"->conv3W1,"PaddingSize"->1],Ramp,
   ConvolutionLayer[256,{3,3},"Biases"->conv3B2,"Weights"->conv3W2,"PaddingSize"->1],Ramp,
   ConvolutionLayer[256,{3,3},"Biases"->conv3B3,"Weights"->conv3W3,"PaddingSize"->1],Ramp,
   PaddingLayer[{{0,0},{0,1},{0,1}}],PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[512,{3,3},"Biases"->conv4B1,"Weights"->conv4W1,"PaddingSize"->1],Ramp,
   ConvolutionLayer[512,{3,3},"Biases"->conv4B2,"Weights"->conv4W2,"PaddingSize"->1],Ramp,
   ConvolutionLayer[512,{3,3},"Biases"->conv4B3,"Weights"->conv4W3,"PaddingSize"->1],Ramp
   },"Input"->{3,300,300}];


net2 = NetChain[{
   PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[512,{3,3},"Biases"->conv5B1,"Weights"->conv5W1,"PaddingSize"->1],Ramp,
   ConvolutionLayer[512,{3,3},"Biases"->conv5B2,"Weights"->conv5W2,"PaddingSize"->1],Ramp,
   ConvolutionLayer[512,{3,3},"Biases"->conv5B3,"Weights"->conv5W3,"PaddingSize"->1],Ramp,
   ConvolutionLayer[1024,{3,3},"Biases"->conv6B,"Weights"->conv6W,"PaddingSize"->6,"Dilation"->6],Ramp,
   ConvolutionLayer[1024,{1,1},"Biases"->conv7B,"Weights"->conv7W],Ramp
}];


SSD[image_]:=( 
   img4d=38+(ImageData[ImageResize[image,{300,300}],Interleaving->False]-.5)*256;
(*img4d=Transpose[Import["c:\\users\\julian\\google drive\\img4d.json"][[1,1]],{2,3,1}];*)
(*img4d=Transpose[Import["c:\\users\\julian\\google drive\\img4d.json"][[1,1]],{2,3,1}];*)
(*
mb11=ConvolutionLayer[64,{3,3},"Biases"->conv1B1,"Weights"->conv1W1,"PaddingSize"->1][img4d];
mb12=Ramp@ConvolutionLayer[64,{3,3},"Biases"->conv1B2,"Weights"->conv1W2,"PaddingSize"->1][Ramp@mb11];
mb1 = PoolingLayer[{2,2},"Stride"->2]@mb12;

mb21=ConvolutionLayer[128,{3,3},"Biases"->conv2B1,"Weights"->conv2W1,"PaddingSize"->1][Ramp@mb1];
mb22=Ramp@ConvolutionLayer[128,{3,3},"Biases"->conv2B2,"Weights"->conv2W2,"PaddingSize"->1][Ramp@mb21];
mb2 = PoolingLayer[{2,2},"Stride"->2]@mb22;

   mb2 = net[img4d];
mb31=ConvolutionLayer[256,{3,3},"Biases"->conv3B1,"Weights"->conv3W1,"PaddingSize"->1][Ramp@mb2];
mb32=Ramp@ConvolutionLayer[256,{3,3},"Biases"->conv3B2,"Weights"->conv3W2,"PaddingSize"->1][Ramp@mb31];
mb33=Ramp@ConvolutionLayer[256,{3,3},"Biases"->conv3B3,"Weights"->conv3W3,"PaddingSize"->1][Ramp@mb32];

*)
   mb43 = net1[img4d,TargetDevice->"GPU"];
(*mb3 = PoolingLayer[{2,2},"Stride"->2]@ArrayPad[mb33,{{0,0},{0,1},{0,1}}];*)
(*
mb41=ConvolutionLayer[512,{3,3},"Biases"->conv4B1,"Weights"->conv4W1,"PaddingSize"->1][Ramp@mb3];
mb42=Ramp@ConvolutionLayer[512,{3,3},"Biases"->conv4B2,"Weights"->conv4W2,"PaddingSize"->1][Ramp@mb41];
mb43=Ramp@ConvolutionLayer[512,{3,3},"Biases"->conv4B3,"Weights"->conv4W3,"PaddingSize"->1][Ramp@mb42];
*)


(*mb4 = PoolingLayer[{2,2},"Stride"->2]@mb43;*)

   c=Sqrt[Total[mb43^2]];
   norm=gamma4*Map[#/c&,mb43];
   tmpmultibox4 = ConvolutionLayer[84,{3,3},"Biases"->block4ClassesB,"Weights"->block4ClassesW,"PaddingSize"->1][norm];
   multibox4=SoftmaxLayer[][Transpose[Partition[tmpmultibox4,21],{1,4,2,3}]];
   locs4 = ConvolutionLayer[16,{3,3},"Biases"->block4LocB,"Weights"->block4LocW,"PaddingSize"->1][norm];
   m4=multibox4[[All,All,All,2;;21]];

   mb7 = net2[ mb43 ];
   tmpmultibox7 = ConvolutionLayer[126,{3,3},"Biases"->block7ClassesB,"Weights"->block7ClassesW,"PaddingSize"->1][mb7];
   multibox7=SoftmaxLayer[][Transpose[Partition[tmpmultibox7,21],{1,4,2,3}]];
   m7=multibox7[[All,All,All,2;;21]];
   locs7 = ConvolutionLayer[24,{3,3},"Biases"->block7LocB,"Weights"->block7LocW,"PaddingSize"->1][mb7];

   {CZPascalClasses[[Position[m4,x_/;x>.15][[All,4]]]],
   CZPascalClasses[[Position[m7,x_/;x>.15][[All,4]]]]}
)


CZDecoder[locs_, probs_,{anchorsx_,anchorsy_,anchorsw_,anchorsh_}]:=( 
(* slocs format A{xywh}HW *)
   slocs=Partition[locs,4];
   rec=Position[probs,x_/;x>.1];
   cx=Map[#+anchorsx[[All,All,1]]&,slocs[[All,1]]*anchorsw*0.1];
   cy=Map[#+anchorsy[[All,All,1]]&,slocs[[All,2]]*anchorsh*0.1];
   w=Exp[slocs[[All,3]]*0.2]*anchorsw;
   h=Exp[slocs[[All,4]]*0.2]*anchorsh;

   MapThread[{#1,#2,300*{{#3-#5/2,1-(#4+#6/2)},{#3+#5/2,1-(#4-#6/2)}}}&,{
      CZPascalClasses[[rec[[All,4]]]],
      Extract[probs,rec],
      Extract[cx,rec[[All,1;;3]]],
      Extract[cy,rec[[All,1;;3]]],
      Extract[w,rec[[All,1;;3]]],
      Extract[h,rec[[All,1;;3]]]
}]
)


CZRawDetectObjects[ img_ ] := (
   SSD[img];
   Join[CZDecoder[locs4,m4,{anchorsx0,anchorsy0,anchorsw0,anchorsh0}]]
)


CZDetectObjects[ img_ ] :=
   Flatten[Map[CZNonMaxSuppression,GatherBy[CZRawDetectObjects[img],#[[1]]&]],1]


CZHighlightObjects[ img_ ] := (
   SSD[img];
   HighlightImage[Image[img4d,Interleaving->False]//ImageAdjust,
      CZDisplayObject/@CZDetectObjects[ img ]]
)

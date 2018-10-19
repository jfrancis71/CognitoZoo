(* ::Package:: *)

SSDFileName="CZModels/SSDVGG512VOC07Plus12FT.hdf";


anchorsx1 = Table[x,{y,8/2,512,8},{x,1,512,8}]/512.;
anchorsy1 = Table[y,{y,8/2,512,8},{x,8/2,512,8}]/512.;
anchorsw1 = {20,31.9374,28.2843,14.1421}/512.;
anchorsh1 = {20,31.9374,14.1421,28.2843}/512.;


anchorsx2 = Table[x,{y,16/2,512,16},{x,16/2,512,16}]/512.;
anchorsy2 = Table[y,{y,16/2,512,16},{x,16/2,512,16}]/512.;
anchorsw2 = {51,82.359,72.1249,36.0624,88.3346,29.4449}/512.;
anchorsh2 = {51,82.359,36.0624,72.1249,29.4449,88.3346}/512.;


anchorsx3 = Table[x,{y,32/2,512,32},{x,32/2,512,32}]/512.;
anchorsy3 = Table[y,{y,32/2,512,32},{x,32/2,512,32}]/512.;
anchorsw3 = {133,169.101,188.09,94.0452,230.363,76.7876}/512.;
anchorsh3 = {133,169.101,94.0452,188.09,76.7876,230.363}/512.;


anchorsx4 = Table[x,{y,64/2,512,64},{x,64/2,512,64}]/512.;
anchorsy4 = Table[y,{y,64/2,512,64},{x,64/2,512,64}]/512.;
anchorsw4 = {215,252.27,304.056,152.028,372.391,124.13}/512.;
anchorsh4 = {215,252.27,152.028,304.056,124.13,372.391}/512;


anchorsx5 = Table[x,{y,128/2,512,128},{x,128/2,512,128}]/512.;
anchorsy5 = Table[y,{y,128/2,512,128},{x,128/2,512,128}]/512.;
anchorsw5 = {296,334.497,418.607,209.304,512.687,170.896}/512.;
anchorsh5 = {296,334.497,209.304,418.607,170.896,512.687}/512.;


anchorsx6 = Table[x,{y,256/2,512,256},{x,256/2,512,256}]/512.;
anchorsy6 = Table[y,{y,256/2,512,256},{x,256/2,512,256}]/512.;
anchorsw6 = {378,416.989,534.573,267.286}/512.;
anchorsh6 = {378,416.989,267.286,534.573}/512.;


anchorsx7 = Table[x,{y,512/2,512,512},{x,512/2,512,512}]/512.;
anchorsy7 = Table[x,{y,512/2,512,512},{x,512/2,512,512}]/512.;
anchorsw7 = {460,499.32,650.538,325.269}/512.;
anchorsh7 = {460,499.32,325.269,650.538}/512.;


(* Caffe has BGR ordering for some obscure reason, because of openCV? *)
conv1W1=Reverse[Import[SSDFileName,{"Datasets","/conv1_1W"}],2]*255;
conv1W2=Import[SSDFileName,{"Datasets","/conv1_2W"}];
conv1B1=Import[SSDFileName,{"Datasets","/conv1_1B"}];
conv1B2=Import[SSDFileName,{"Datasets","/conv1_2B"}];

conv2W1=Import[SSDFileName,{"Datasets","/conv2_1W"}];
conv2W2=Import[SSDFileName,{"Datasets","/conv2_2W"}];
conv2B1=Import[SSDFileName,{"Datasets","/conv2_1B"}];
conv2B2=Import[SSDFileName,{"Datasets","/conv2_2B"}];

conv3W1=Import[SSDFileName,{"Datasets","/conv3_1W"}];
conv3W2=Import[SSDFileName,{"Datasets","/conv3_2W"}];
conv3W3=Import[SSDFileName,{"Datasets","/conv3_3W"}];
conv3B1=Import[SSDFileName,{"Datasets","/conv3_1B"}];
conv3B2=Import[SSDFileName,{"Datasets","/conv3_2B"}];
conv3B3=Import[SSDFileName,{"Datasets","/conv3_3B"}];

conv4W1=Import[SSDFileName,{"Datasets","/conv4_1W"}];
conv4W2=Import[SSDFileName,{"Datasets","/conv4_2W"}];
conv4W3=Import[SSDFileName,{"Datasets","/conv4_3W"}];
conv4B1=Import[SSDFileName,{"Datasets","/conv4_1B"}];
conv4B2=Import[SSDFileName,{"Datasets","/conv4_2B"}];
conv4B3=Import[SSDFileName,{"Datasets","/conv4_3B"}];


conv5W1=Import[SSDFileName,{"Datasets","/conv5_1W"}];
conv5W2=Import[SSDFileName,{"Datasets","/conv5_2W"}];
conv5W3=Import[SSDFileName,{"Datasets","/conv5_3W"}];
conv5B1=Import[SSDFileName,{"Datasets","/conv5_1B"}];
conv5B2=Import[SSDFileName,{"Datasets","/conv5_2B"}];
conv5B3=Import[SSDFileName,{"Datasets","/conv5_3B"}];


conv6W=Import[SSDFileName,{"Datasets","/conv6_W"}];
conv6B=Import[SSDFileName,{"Datasets","/conv6_B"}];


conv7W=Import[SSDFileName,{"Datasets","/conv7_W"}];
conv7B=Import[SSDFileName,{"Datasets","/conv7_B"}];


conv8W1=Import[SSDFileName,{"Datasets","/conv8_1W"}];
conv8W2=Import[SSDFileName,{"Datasets","/conv8_2W"}];
conv8B1=Import[SSDFileName,{"Datasets","/conv8_1B"}];
conv8B2=Import[SSDFileName,{"Datasets","/conv8_2B"}];


conv9W1=Import[SSDFileName,{"Datasets","/conv9_1W"}];
conv9W2=Import[SSDFileName,{"Datasets","/conv9_2W"}];
conv9B1=Import[SSDFileName,{"Datasets","/conv9_1B"}];
conv9B2=Import[SSDFileName,{"Datasets","/conv9_2B"}];


conv10W1=Import[SSDFileName,{"Datasets","/conv10_1W"}];
conv10W2=Import[SSDFileName,{"Datasets","/conv10_2W"}];
conv10B1=Import[SSDFileName,{"Datasets","/conv10_1B"}];
conv10B2=Import[SSDFileName,{"Datasets","/conv10_2B"}];


conv11W1=Import[SSDFileName,{"Datasets","/conv11_1W"}];
conv11W2=Import[SSDFileName,{"Datasets","/conv11_2W"}];
conv11B1=Import[SSDFileName,{"Datasets","/conv11_1B"}];
conv11B2=Import[SSDFileName,{"Datasets","/conv11_2B"}];


conv12W1=Import[SSDFileName,{"Datasets","/conv12_1W"}];
conv12W2=Import[SSDFileName,{"Datasets","/conv12_2W"}];
conv12B1=Import[SSDFileName,{"Datasets","/conv12_1B"}];
conv12B2=Import[SSDFileName,{"Datasets","/conv12_2B"}];


block4ClassesW = Import[SSDFileName,{"Datasets","/block4_classes_W"}];
block4ClassesB = Import[SSDFileName,{"Datasets","/block4_classes_B"}];
block4LocW = Import[SSDFileName,{"Datasets","/block4_loc_W"}];
block4LocB = Import[SSDFileName,{"Datasets","/block4_loc_B"}];


block7ClassesW = Import[SSDFileName,{"Datasets","/block7_classes_W"}];
block7ClassesB = Import[SSDFileName,{"Datasets","/block7_classes_B"}];
block7LocW = Import[SSDFileName,{"Datasets","/block7_loc_W"}];
block7LocB = Import[SSDFileName,{"Datasets","/block7_loc_B"}];


block8ClassesW = Import[SSDFileName,{"Datasets","/block8_classes_W"}];
block8ClassesB = Import[SSDFileName,{"Datasets","/block8_classes_B"}];
block8LocW = Import[SSDFileName,{"Datasets","/block8_loc_W"}];
block8LocB = Import[SSDFileName,{"Datasets","/block8_loc_B"}];


block9ClassesW = Import[SSDFileName,{"Datasets","/block9_classes_W"}];
block9ClassesB = Import[SSDFileName,{"Datasets","/block9_classes_B"}];
block9LocW = Import[SSDFileName,{"Datasets","/block9_loc_W"}];
block9LocB = Import[SSDFileName,{"Datasets","/block9_loc_B"}];


block10ClassesW = Import[SSDFileName,{"Datasets","/block10_classes_W"}];
block10ClassesB = Import[SSDFileName,{"Datasets","/block10_classes_B"}];
block10LocW = Import[SSDFileName,{"Datasets","/block10_loc_W"}];
block10LocB = Import[SSDFileName,{"Datasets","/block10_loc_B"}];


block11ClassesW = Import[SSDFileName,{"Datasets","/block11_classes_W"}];
block11ClassesB = Import[SSDFileName,{"Datasets","/block11_classes_B"}];
block11LocW = Import[SSDFileName,{"Datasets","/block11_loc_W"}];
block11LocB = Import[SSDFileName,{"Datasets","/block11_loc_B"}];


block12ClassesW = Import[SSDFileName,{"Datasets","/block12_classes_W"}];
block12ClassesB = Import[SSDFileName,{"Datasets","/block12_classes_B"}];
block12LocW = Import[SSDFileName,{"Datasets","/block12_loc_W"}];
block12LocB = Import[SSDFileName,{"Datasets","/block12_loc_B"}];


ngamma4c = Import[SSDFileName,{"Datasets","/conv4_3_norm"}];
ngamma4=Table[ngamma4c[[c]],{c,1,512},{64},{64}];


initConvNet = NetReplacePart[ ssdNet,{
   {1,1,"blockNet4","conv1a",1,"Weights"}->conv1W1,
   {1,1,"blockNet4","conv1a",1,"Biases"}->conv1B1,   
   {1,1,"blockNet4","conv1b",1,"Weights"}->conv1W2,
   {1,1,"blockNet4","conv1b",1,"Biases"}->conv1B2,   
   {1,1,"blockNet4","conv2a",1,"Weights"}->conv2W1,
   {1,1,"blockNet4","conv2a",1,"Biases"}->conv2B1,   
   {1,1,"blockNet4","conv2b",1,"Weights"}->conv2W2,
   {1,1,"blockNet4","conv2b",1,"Biases"}->conv2B2,   
   {1,1,"blockNet4","conv3a",1,"Weights"}->conv3W1,
   {1,1,"blockNet4","conv3a",1,"Biases"}->conv3B1,
   {1,1,"blockNet4","conv3b",1,"Weights"}->conv3W2,
   {1,1,"blockNet4","conv3b",1,"Biases"}->conv3B2,   
   {1,1,"blockNet4","conv3c",1,"Weights"}->conv3W3,
   {1,1,"blockNet4","conv3c",1,"Biases"}->conv3B3,
   {1,1,"blockNet4","conv4a",1,"Weights"}->conv4W1,
   {1,1,"blockNet4","conv4a",1,"Biases"}->conv4B1,
   {1,1,"blockNet4","conv4b",1,"Weights"}->conv4W2,
   {1,1,"blockNet4","conv4b",1,"Biases"}->conv4B2,
   {1,1,"blockNet4","conv4c",1,"Weights"}->conv4W3,
   {1,1,"blockNet4","conv4c",1,"Biases"}->conv4B3,
   
   {1,1,"blockNet7","conv5a",1,"Weights"}->conv5W1,
   {1,1,"blockNet7","conv5a",1,"Biases"}->conv5B1,
   {1,1,"blockNet7","conv5b",1,"Weights"}->conv5W2,
   {1,1,"blockNet7","conv5b",1,"Biases"}->conv5B2,
   {1,1,"blockNet7","conv5c",1,"Weights"}->conv5W3,
   {1,1,"blockNet7","conv5c",1,"Biases"}->conv5B3,
   {1,1,"blockNet7","conv6",1,"Weights"}->conv6W,
   {1,1,"blockNet7","conv6",1,"Biases"}->conv6B,
   {1,1,"blockNet7","conv7",1,"Weights"}->conv7W,
   {1,1,"blockNet7","conv7",1,"Biases"}->conv7B,
   
   {1,1,"blockNet8","conv8a",1,"Weights"}->conv8W1,
   {1,1,"blockNet8","conv8a",1,"Biases"}->conv8B1,
   {1,1,"blockNet8","conv8b",1,"Weights"}->conv8W2,
   {1,1,"blockNet8","conv8b",1,"Biases"}->conv8B2,
   
   {1,1,"blockNet9","conv9a",1,"Weights"}->conv9W1,
   {1,1,"blockNet9","conv9a",1,"Biases"}->conv9B1,
   {1,1,"blockNet9","conv9b",1,"Weights"}->conv9W2,
   {1,1,"blockNet9","conv9b",1,"Biases"}->conv9B2,

   {1,1,"blockNet10","conv10a",1,"Weights"}->conv10W1,
   {1,1,"blockNet10","conv10a",1,"Biases"}->conv10B1,
   {1,1,"blockNet10","conv10b",1,"Weights"}->conv10W2,
   {1,1,"blockNet10","conv10b",1,"Biases"}->conv10B2,

   {1,1,"blockNet11","conv11a",1,"Weights"}->conv11W1,
   {1,1,"blockNet11","conv11a",1,"Biases"}->conv11B1,
   {1,1,"blockNet11","conv11b",1,"Weights"}->conv11W2,
   {1,1,"blockNet11","conv11b",1,"Biases"}->conv11B2,

   {1,1,"blockNet12","conv12a",1,"Weights"}->conv12W1,
   {1,1,"blockNet12","conv12a",1,"Biases"}->conv12B1,
   {1,1,"blockNet12","conv12b",1,"Weights"}->conv12W2,
   {1,1,"blockNet12","conv12b",1,"Biases"}->conv12B2,
   
   
   
   {1,1,"multiboxLayer1","multiboxClasses",1,"Weights"}->block4ClassesW,
   {1,1,"multiboxLayer1","multiboxClasses",1,"Biases"}->block4ClassesB,
   {1,1,"multiboxLayer2","multiboxClasses",1,"Weights"}->block7ClassesW,
   {1,1,"multiboxLayer2","multiboxClasses",1,"Biases"}->block7ClassesB,
   {1,1,"multiboxLayer3","multiboxClasses",1,"Weights"}->block8ClassesW,
   {1,1,"multiboxLayer3","multiboxClasses",1,"Biases"}->block8ClassesB,
   {1,1,"multiboxLayer4","multiboxClasses",1,"Weights"}->block9ClassesW,
   {1,1,"multiboxLayer4","multiboxClasses",1,"Biases"}->block9ClassesB,
   {1,1,"multiboxLayer5","multiboxClasses",1,"Weights"}->block10ClassesW,
   {1,1,"multiboxLayer5","multiboxClasses",1,"Biases"}->block10ClassesB,
   {1,1,"multiboxLayer6","multiboxClasses",1,"Weights"}->block11ClassesW,
   {1,1,"multiboxLayer6","multiboxClasses",1,"Biases"}->block11ClassesB,
   {1,1,"multiboxLayer7","multiboxClasses",1,"Weights"}->block12ClassesW,
   {1,1,"multiboxLayer7","multiboxClasses",1,"Biases"}->block12ClassesB,

   {1,1,"multiboxLayer1","multiboxLocs","convloc",1,"Weights"}->block4LocW,
   {1,1,"multiboxLayer1","multiboxLocs","convloc",1,"Biases"}->block4LocB,
   {1,1,"multiboxLayer2","multiboxLocs","convloc",1,"Weights"}->block7LocW,
   {1,1,"multiboxLayer2","multiboxLocs","convloc",1,"Biases"}->block7LocB,
   {1,1,"multiboxLayer3","multiboxLocs","convloc",1,"Weights"}->block8LocW,
   {1,1,"multiboxLayer3","multiboxLocs","convloc",1,"Biases"}->block8LocB,
   {1,1,"multiboxLayer4","multiboxLocs","convloc",1,"Weights"}->block9LocW,
   {1,1,"multiboxLayer4","multiboxLocs","convloc",1,"Biases"}->block9LocB,
   {1,1,"multiboxLayer5","multiboxLocs","convloc",1,"Weights"}->block10LocW,
   {1,1,"multiboxLayer5","multiboxLocs","convloc",1,"Biases"}->block10LocB,
   {1,1,"multiboxLayer6","multiboxLocs","convloc",1,"Weights"}->block11LocW,
   {1,1,"multiboxLayer6","multiboxLocs","convloc",1,"Biases"}->block11LocB,
   {1,1,"multiboxLayer7","multiboxLocs","convloc",1,"Weights"}->block12LocW,
   {1,1,"multiboxLayer7","multiboxLocs","convloc",1,"Biases"}->block12LocB,

      
   {1,1,"multiboxLayer1","channelNorm1",6,"Scaling"}->ngamma4
}];


trainedModel = NetReplacePart[ initConvNet,{
   {2,"cx",2,"Scaling"}->.1*Flatten[MapThread[ ConstantArray[ #1, #2 ]&, {
      { anchorsw1, anchorsw2, anchorsw3, anchorsw4, anchorsw5, anchorsw6, anchorsw7 },
      { {64,64}, {32,32}, {16,16}, {8,8}, {4,4}, {2,2}, {1,1} } } ]],
   {2,"cx",3,"Biases"}->Flatten@MapThread[ Transpose[ ConstantArray[ #1,{ #2 } ], {3,1,2} ]&,
      { { anchorsx1, anchorsx2, anchorsx3, anchorsx4, anchorsx5, anchorsx6, anchorsx7 }, 
      { 4,6,6,6,6,4,4 } } ],
   {2,"cy",2,"Scaling"}->.1*Flatten[MapThread[ ConstantArray[ #1, #2 ]&, {
      { anchorsh1, anchorsh2, anchorsh3, anchorsh4, anchorsh5, anchorsh6, anchorsh7 },
      { {64,64}, {32,32}, {16,16}, {8,8}, {4,4}, {2,2}, {1,1} } } ]],
   {2,"cy",3,"Biases"}->Flatten@MapThread[ Transpose[ ConstantArray[ #1,{ #2 } ], {3,1,2} ]&,
      { { anchorsy1, anchorsy2, anchorsy3, anchorsy4, anchorsy5, anchorsy6, anchorsy7 }, 
      { 4,6,6,6,6,4,4 } } ],
   {2,"width",3,"Scaling"}->Flatten[MapThread[ ConstantArray[ #1, #2 ]&, {
      { anchorsw1, anchorsw2, anchorsw3, anchorsw4, anchorsw5, anchorsw6, anchorsw7 },
      { {64,64}, {32,32}, {16,16}, {8,8}, {4,4}, {2,2}, {1,1} } } ]],
   {2,"height",3,"Scaling"}->Flatten[MapThread[ ConstantArray[ #1, #2 ]&, {
      { anchorsh1, anchorsh2, anchorsh3, anchorsh4, anchorsh5, anchorsh6, anchorsh7 },
      { {64,64}, {32,32}, {16,16}, {8,8}, {4,4}, {2,2}, {1,1} } } ]]
}];


(* Exporting code and setting permissions:

   CloudExport[ trainedModel, "WLNET","SSDVGG512VOC07Plus12FT.wlnet" ]
   Options[CloudObject["SSDVGG512VOC07Plus12FT.wlnet"],Permissions]
   SetOptions[CloudObject["SSDVGG512VOC07Plus12FT.wlnet"], Permissions\[Rule]"Public"]
*)

(* ::Package:: *)

SSDFileName="CZModels/SSDVGG300PascalVOCReference20180920.hdf";


anchorsx1 = Table[2*x/75,{y,1,38},{x,.5,37.5}];
anchorsy1 = Table[2*y/75,{y,.5,37.5},{x,1,38}];
anchorsw1 = {30/300,42/300,42/300,21/300};
anchorsh1 = {30/300,42/300,21/300,42/300};


anchorsx2 = Table[4*x/75,{y,1,19},{x,.5,18.5}];
anchorsy2 = Table[4*y/75,{y,.5,18.5},{x,1,19}];
anchorsw2 = {60/300,81/300,84/300,42/300,103/300,34/300};
anchorsh2 = {60/300,81/300,42/300,84/300,34/300,103/300};


anchorsx3 = Table[8*x/75,{y,1,10},{x,.5,9.5}];
anchorsy3 = Table[8*y/75,{y,.5,9.5},{x,1,10}];
anchorsw3 = {111,134,156,78,192,64}/300;
anchorsh3 = {111,134,78,156,64,192}/300;


anchorsx4 = Table[16*x/75,{y,1,5},{x,.5,4.5}];
anchorsy4 = Table[16*y/75,{y,.5,4.5},{x,1,5}];
anchorsw4 = {162,185,229,114,280,93}/300;
anchorsh4 = {162,185,114,229,93,280}/300;


anchorsx5 = Table[x/3,{y,1,3},{x,.5,2.5}];
anchorsy5 = Table[y/3,{y,.5,2.5},{x,1,3}];
anchorsw5 = {213,237,301,150}/300;
anchorsh5 = {213,237,150,301}/300;


anchorsx6 = Table[x/2,{y,1,1},{x,1,1}];
anchorsy6 = Table[y/2,{y,1,1},{x,1,1}];
anchorsw6 = {264,288,373,186}/300;
anchorsh6 = {264,288,186,373}/300;


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


ngamma4c = Import[SSDFileName,{"Datasets","/conv4_3_norm"}];
ngamma4=Table[ngamma4c[[c]],{c,1,512},{38},{38}];


trainedModel = NetReplacePart[ ssdNet,{
   {1,"blockNet4","conv1a",1,"Weights"}->conv1W1,
   {1,"blockNet4","conv1a",1,"Biases"}->conv1B1,   
   {1,"blockNet4","conv1b",1,"Weights"}->conv1W2,
   {1,"blockNet4","conv1b",1,"Biases"}->conv1B2,   
   {1,"blockNet4","conv2a",1,"Weights"}->conv2W1,
   {1,"blockNet4","conv2a",1,"Biases"}->conv2B1,   
   {1,"blockNet4","conv2b",1,"Weights"}->conv2W2,
   {1,"blockNet4","conv2b",1,"Biases"}->conv2B2,   
   {1,"blockNet4","conv3a",1,"Weights"}->conv3W1,
   {1,"blockNet4","conv3a",1,"Biases"}->conv3B1,
   {1,"blockNet4","conv3b",1,"Weights"}->conv3W2,
   {1,"blockNet4","conv3b",1,"Biases"}->conv3B2,   
   {1,"blockNet4","conv3c",1,"Weights"}->conv3W3,
   {1,"blockNet4","conv3c",1,"Biases"}->conv3B3,
   {1,"blockNet4","conv4a",1,"Weights"}->conv4W1,
   {1,"blockNet4","conv4a",1,"Biases"}->conv4B1,
   {1,"blockNet4","conv4b",1,"Weights"}->conv4W2,
   {1,"blockNet4","conv4b",1,"Biases"}->conv4B2,
   {1,"blockNet4","conv4c",1,"Weights"}->conv4W3,
   {1,"blockNet4","conv4c",1,"Biases"}->conv4B3,
   
   {1,"blockNet7","conv5a",1,"Weights"}->conv5W1,
   {1,"blockNet7","conv5a",1,"Biases"}->conv5B1,
   {1,"blockNet7","conv5b",1,"Weights"}->conv5W2,
   {1,"blockNet7","conv5b",1,"Biases"}->conv5B2,
   {1,"blockNet7","conv5c",1,"Weights"}->conv5W3,
   {1,"blockNet7","conv5c",1,"Biases"}->conv5B3,
   {1,"blockNet7","conv6",1,"Weights"}->conv6W,
   {1,"blockNet7","conv6",1,"Biases"}->conv6B,
   {1,"blockNet7","conv7",1,"Weights"}->conv7W,
   {1,"blockNet7","conv7",1,"Biases"}->conv7B,
   
   {1,"blockNet8","conv8a",1,"Weights"}->conv8W1,
   {1,"blockNet8","conv8a",1,"Biases"}->conv8B1,
   {1,"blockNet8","conv8b",1,"Weights"}->conv8W2,
   {1,"blockNet8","conv8b",1,"Biases"}->conv8B2,
   
   {1,"blockNet9","conv9a",1,"Weights"}->conv9W1,
   {1,"blockNet9","conv9a",1,"Biases"}->conv9B1,
   {1,"blockNet9","conv9b",1,"Weights"}->conv9W2,
   {1,"blockNet9","conv9b",1,"Biases"}->conv9B2,

   {1,"blockNet10","conv10a",1,"Weights"}->conv10W1,
   {1,"blockNet10","conv10a",1,"Biases"}->conv10B1,
   {1,"blockNet10","conv10b",1,"Weights"}->conv10W2,
   {1,"blockNet10","conv10b",1,"Biases"}->conv10B2,

   {1,"blockNet11","conv11a",1,"Weights"}->conv11W1,
   {1,"blockNet11","conv11a",1,"Biases"}->conv11B1,
   {1,"blockNet11","conv11b",1,"Weights"}->conv11W2,
   {1,"blockNet11","conv11b",1,"Biases"}->conv11B2,
   
   {1,"multiboxLayer1","multiboxClasses",1,"Weights"}->block4ClassesW,
   {1,"multiboxLayer1","multiboxClasses",1,"Biases"}->block4ClassesB,
   {1,"multiboxLayer2","multiboxClasses",1,"Weights"}->block7ClassesW,
   {1,"multiboxLayer2","multiboxClasses",1,"Biases"}->block7ClassesB,
   {1,"multiboxLayer3","multiboxClasses",1,"Weights"}->block8ClassesW,
   {1,"multiboxLayer3","multiboxClasses",1,"Biases"}->block8ClassesB,
   {1,"multiboxLayer4","multiboxClasses",1,"Weights"}->block9ClassesW,
   {1,"multiboxLayer4","multiboxClasses",1,"Biases"}->block9ClassesB,
   {1,"multiboxLayer5","multiboxClasses",1,"Weights"}->block10ClassesW,
   {1,"multiboxLayer5","multiboxClasses",1,"Biases"}->block10ClassesB,
   {1,"multiboxLayer6","multiboxClasses",1,"Weights"}->block11ClassesW,
   {1,"multiboxLayer6","multiboxClasses",1,"Biases"}->block11ClassesB,
   
   {1,"multiboxLayer1","multiboxLocs","convloc",1,"Weights"}->block4LocW,
   {1,"multiboxLayer1","multiboxLocs","convloc",1,"Biases"}->block4LocB,
   {1,"multiboxLayer2","multiboxLocs","convloc",1,"Weights"}->block7LocW,
   {1,"multiboxLayer2","multiboxLocs","convloc",1,"Biases"}->block7LocB,
   {1,"multiboxLayer3","multiboxLocs","convloc",1,"Weights"}->block8LocW,
   {1,"multiboxLayer3","multiboxLocs","convloc",1,"Biases"}->block8LocB,
   {1,"multiboxLayer4","multiboxLocs","convloc",1,"Weights"}->block9LocW,
   {1,"multiboxLayer4","multiboxLocs","convloc",1,"Biases"}->block9LocB,
   {1,"multiboxLayer5","multiboxLocs","convloc",1,"Weights"}->block10LocW,
   {1,"multiboxLayer5","multiboxLocs","convloc",1,"Biases"}->block10LocB,
   {1,"multiboxLayer6","multiboxLocs","convloc",1,"Weights"}->block11LocW,
   {1,"multiboxLayer6","multiboxLocs","convloc",1,"Biases"}->block11LocB,
      
   {1,"multiboxLayer1","channelNorm1",6,"Scaling"}->ngamma4,
   {1,"multiboxLayer1","multiboxLocs","cx",2,"Scaling"}->Table[0.1*anchorsw1[[b]],{b,1,4},{38},{38}],
   {1,"multiboxLayer1","multiboxLocs","cx",3,"Biases"}->ConstantArray[anchorsx1,{4}],   
   {1,"multiboxLayer1","multiboxLocs","cy",2,"Scaling"}->Table[0.1*anchorsh1[[b]],{b,1,4},{38},{38}],
   {1,"multiboxLayer1","multiboxLocs","cy",3,"Biases"}->ConstantArray[anchorsy1,{4}],   
   {1,"multiboxLayer1","multiboxLocs","width",3,"Scaling"}->Table[anchorsw1[[b]],{b,1,4},{38},{38}],
   {1,"multiboxLayer1","multiboxLocs","height",3,"Scaling"}->Table[anchorsh1[[b]],{b,1,4},{38},{38}],

   {1,"multiboxLayer2","multiboxLocs","cx",2,"Scaling"}->Table[0.1*anchorsw2[[b]],{b,1,6},{19},{19}],
   {1,"multiboxLayer2","multiboxLocs","cx",3,"Biases"}->ConstantArray[anchorsx2,{6}],   
   {1,"multiboxLayer2","multiboxLocs","cy",2,"Scaling"}->Table[0.1*anchorsh2[[b]],{b,1,6},{19},{19}],
   {1,"multiboxLayer2","multiboxLocs","cy",3,"Biases"}->ConstantArray[anchorsy2,{6}],   
   {1,"multiboxLayer2","multiboxLocs","width",3,"Scaling"}->Table[anchorsw2[[b]],{b,1,6},{19},{19}],
   {1,"multiboxLayer2","multiboxLocs","height",3,"Scaling"}->Table[anchorsh2[[b]],{b,1,6},{19},{19}],
   
   {1,"multiboxLayer3","multiboxLocs","cx",2,"Scaling"}->Table[0.1*anchorsw3[[b]],{b,1,6},{10},{10}],
   {1,"multiboxLayer3","multiboxLocs","cx",3,"Biases"}->ConstantArray[anchorsx3,{6}],   
   {1,"multiboxLayer3","multiboxLocs","cy",2,"Scaling"}->Table[0.1*anchorsh3[[b]],{b,1,6},{10},{10}],
   {1,"multiboxLayer3","multiboxLocs","cy",3,"Biases"}->ConstantArray[anchorsy3,{6}],   
   {1,"multiboxLayer3","multiboxLocs","width",3,"Scaling"}->Table[anchorsw3[[b]],{b,1,6},{10},{10}],
   {1,"multiboxLayer3","multiboxLocs","height",3,"Scaling"}->Table[anchorsh3[[b]],{b,1,6},{10},{10}],
   
   {1,"multiboxLayer4","multiboxLocs","cx",2,"Scaling"}->Table[0.1*anchorsw4[[b]],{b,1,6},{5},{5}],
   {1,"multiboxLayer4","multiboxLocs","cx",3,"Biases"}->ConstantArray[anchorsx4,{6}],   
   {1,"multiboxLayer4","multiboxLocs","cy",2,"Scaling"}->Table[0.1*anchorsh4[[b]],{b,1,6},{5},{5}],
   {1,"multiboxLayer4","multiboxLocs","cy",3,"Biases"}->ConstantArray[anchorsy4,{6}],   
   {1,"multiboxLayer4","multiboxLocs","width",3,"Scaling"}->Table[anchorsw4[[b]],{b,1,6},{5},{5}],
   {1,"multiboxLayer4","multiboxLocs","height",3,"Scaling"}->Table[anchorsh4[[b]],{b,1,6},{5},{5}],

   {1,"multiboxLayer5","multiboxLocs","cx",2,"Scaling"}->Table[0.1*anchorsw5[[b]],{b,1,4},{3},{3}],
   {1,"multiboxLayer5","multiboxLocs","cx",3,"Biases"}->ConstantArray[anchorsx5,{4}],   
   {1,"multiboxLayer5","multiboxLocs","cy",2,"Scaling"}->Table[0.1*anchorsh5[[b]],{b,1,4},{3},{3}],
   {1,"multiboxLayer5","multiboxLocs","cy",3,"Biases"}->ConstantArray[anchorsy5,{4}],   
   {1,"multiboxLayer5","multiboxLocs","width",3,"Scaling"}->Table[anchorsw5[[b]],{b,1,4},{3},{3}],
   {1,"multiboxLayer5","multiboxLocs","height",3,"Scaling"}->Table[anchorsh5[[b]],{b,1,4},{3},{3}],

   {1,"multiboxLayer6","multiboxLocs","cx",2,"Scaling"}->Table[0.1*anchorsw6[[b]],{b,1,4},{1},{1}],
   {1,"multiboxLayer6","multiboxLocs","cx",3,"Biases"}->ConstantArray[anchorsx6,{4}],   
   {1,"multiboxLayer6","multiboxLocs","cy",2,"Scaling"}->Table[0.1*anchorsh6[[b]],{b,1,4},{1},{1}],
   {1,"multiboxLayer6","multiboxLocs","cy",3,"Biases"}->ConstantArray[anchorsy6,{4}],   
   {1,"multiboxLayer6","multiboxLocs","width",3,"Scaling"}->Table[anchorsw6[[b]],{b,1,4},{1},{1}],
   {1,"multiboxLayer6","multiboxLocs","height",3,"Scaling"}->Table[anchorsh6[[b]],{b,1,4},{1},{1}]
}];


(* Exporting code and setting permissions:

   CloudExport[ trainedModel, "WLNET","SSDVGG300PascalVOCReference20180920.wlnet" ]
   Options[CloudObject["SSDVGG300PascalVOCReference20180920.wlnet"],Permissions]
   SetOptions[CloudObject["SSDVGG300PascalVOCReference20180920.wlnet"], Permissions\[Rule]"Public"]
*)

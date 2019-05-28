(* ::Package:: *)

Relu6 = ElementwiseLayer[Min[Ramp[#],6]&];


mobileUnit[ depthwise_, channels_, kernelDims_, stride_, padding_, activationFn_ ] := NetChain[{
   "conv2d"->ConvolutionLayer[channels,kernelDims,"Stride"->stride,"PaddingSize"->padding,"ChannelGroups"->If[depthwise,channels,1]],
   "BatchNorm"->BatchNormalizationLayer[], Sequence[ If[activationFn===Nothing,Nothing,"ActivationFn"->activationFn ] ] } ];


mobileNetBlock[depthwiseStride_,depthWisePadding_,expandChannels_, projectChannels_] := NetChain[{
   "expand"->mobileUnit[ False, expandChannels, {1,1}, 1, 0, Relu6 ],
   "depthwise"->mobileUnit[ True, expandChannels, {3,3}, depthwiseStride, depthWisePadding, Relu6 ],
   "project"->mobileUnit[ False, projectChannels, {1,1}, 1, 0, Nothing ]
}];


invResMobileNetBlock[expandChannels_, projectChannels_] :=
   NetGraph[{1->mobileNetBlock[ 1, 1, expandChannels, projectChannels ],2->ThreadingLayer[Plus]},{{1,NetPort["Input"]}->2}];


ssdMobileNetFeatureExtractor = NetGraph[{

   "Conv"->mobileUnit[ False, 32, {3,3}, 2, {{0,1},{0,1}}, Relu6 ], 
   "expanded_conv/depthwise"->mobileUnit[ True, 32, {3,3}, 1, 1, Relu6 ], 
   "expanded_conv/project"->mobileUnit[False, 16, {1,1}, 1, 0, Nothing ],
   
   "expanded_conv_1"->mobileNetBlock[ 2, {{0,1},{0,1}}, 6*16, 24 ],
   "expanded_conv_2"->invResMobileNetBlock[ 6*24, 24 ],
   
   "expanded_conv_3"->mobileNetBlock[ 2, 1, 6*24, 32 ],
   "expanded_conv_4"->invResMobileNetBlock[ 6*32, 32 ],
   "expanded_conv_5"->invResMobileNetBlock[ 6*32, 32 ],
   
   "expanded_conv_6"->mobileNetBlock[ 2, {{0,1},{0,1}}, 6*32, 64 ],
   "expanded_conv_7"->invResMobileNetBlock[ 6*64, 64 ],
   "expanded_conv_8"->invResMobileNetBlock[ 6*64, 64 ],
   "expanded_conv_9"->invResMobileNetBlock[ 6*64, 64 ],

   "expanded_conv_10"->mobileNetBlock[ 1, 1, 6*64, 96 ],
   "expanded_conv_11"->invResMobileNetBlock[ 6*96, 96 ],
   "expanded_conv_12"->invResMobileNetBlock[ 6*96, 96 ],

   "expanded_conv_13/expand"->mobileUnit[ False, 6*96, {1,1}, 1, 0, Relu6 ],(* port multibox0 *)
   (* hence we construct seperately this subnet at this point (as opposed to using mobileNetBlock) *)
   "expanded_conv_13/depthwise"->mobileUnit[ True, 6*96, {3,3}, 2, 1, Relu6 ],
   "expanded_conv_13/project"->mobileUnit[ False, 160, {1,1}, 1, 0, Nothing ],

   "expanded_conv_14"->invResMobileNetBlock[ 6*160, 160 ],
   "expanded_conv_15"->invResMobileNetBlock[ 6*160, 160 ],
   
   "expanded_conv_16"->mobileNetBlock[ 1, 1, 6*160, 320 ],
   "Conv_1"->mobileUnit[ False, 1280, {1,1}, 1, 0, Relu6 ], (* port multibox1 *)
   "layer_19_1_Conv2d_2_1x1_256"->mobileUnit[ False, 256, {1,1}, 1, 0, Relu6 ],
   "layer_19_2_Conv2d_2_3x3_s2_512_depthwise"->mobileUnit[ True, 256, {3,3}, 2, { {0,1}, {0,1} }, Relu6 ],
   "layer_19_2_Conv2d_2_3x3_s2_512"->mobileUnit[ False, 512, {1,1}, 1, 0, Relu6 ],(* port multibox2 *)
   "layer_19_1_Conv2d_3_1x1_128"->mobileUnit[ False, 128, {1,1}, 1, 0, Relu6 ],
   "layer_19_2_Conv2d_3_3x3_s2_256_depthwise"->mobileUnit[ True, 128, {3,3}, 2, 1, Relu6 ],
   "layer_19_2_Conv2d_3_3x3_s2_256"->mobileUnit[ False, 256, {1,1}, 1, 0, Relu6 ],(* port multibox3 *)
   "layer_19_1_Conv2d_4_1x1_128"->mobileUnit[ False, 128, {1,1}, 1, 0, Relu6 ],
   "layer_19_2_Conv2d_4_3x3_s2_256_depthwise"->mobileUnit[ True, 128, {3,3}, 2, 1,Relu6 ],
   "layer_19_2_Conv2d_4_3x3_s2_256"->mobileUnit[ False, 256, {1,1}, 1, 0, Relu6 ],(* port multibox4 *)
   "layer_19_1_Conv2d_5_1x1_64"->mobileUnit[ False, 64, {1,1}, 1, 0, Relu6 ],
   "layer_19_2_Conv2d_5_3x3_s2_128_depthwise"->mobileUnit[ True, 64, {3,3}, 2, {{0,1},{0,1}}, Relu6 ],
   "layer_19_2_Conv2d_5_3x3_s2_128"->mobileUnit[ False, 128, {1,1}, 1, 0, Relu6 ]},(* port multibox5 *)

{
   "Conv"->"expanded_conv/depthwise"->"expanded_conv/project"->"expanded_conv_1"->"expanded_conv_2"->
   "expanded_conv_3"->"expanded_conv_4"->"expanded_conv_5"->"expanded_conv_6"->"expanded_conv_7"->
   "expanded_conv_8"->"expanded_conv_9"->"expanded_conv_10"->"expanded_conv_11"->"expanded_conv_12"->
(* Need to bolt conv_13 back together *)
   "expanded_conv_13/expand"->"expanded_conv_13/depthwise"->"expanded_conv_13/project"->
   
   "expanded_conv_14"->"expanded_conv_15"->"expanded_conv_16"->
   "Conv_1"->"layer_19_1_Conv2d_2_1x1_256"->"layer_19_2_Conv2d_2_3x3_s2_512_depthwise"->
   "layer_19_2_Conv2d_2_3x3_s2_512"->"layer_19_1_Conv2d_3_1x1_128"->
   "layer_19_2_Conv2d_3_3x3_s2_256_depthwise"->"layer_19_2_Conv2d_3_3x3_s2_256"->
   "layer_19_1_Conv2d_4_1x1_128"->"layer_19_2_Conv2d_4_3x3_s2_256_depthwise"->
   "layer_19_2_Conv2d_4_3x3_s2_256"->"layer_19_1_Conv2d_5_1x1_64"->
   "layer_19_2_Conv2d_5_3x3_s2_128_depthwise"->"layer_19_2_Conv2d_5_3x3_s2_128",

(*\[NonBreakingSpace]Need to export the feature ports *)
   "expanded_conv_13/expand"->NetPort["multiBox0"],
   "Conv_1"->NetPort["multiBox1"],
   "layer_19_2_Conv2d_2_3x3_s2_512"->NetPort["multiBox2"],
   "layer_19_2_Conv2d_3_3x3_s2_256"->NetPort["multiBox3"],
   "layer_19_2_Conv2d_4_3x3_s2_256"->NetPort["multiBox4"],
   "layer_19_2_Conv2d_5_3x3_s2_128"->NetPort["multiBox5"]

}];


multiBoxLayer[ anchors_, layer_String, size_ ] := NetGraph[{
   "ClassPredictor"->ConvolutionLayer[80*anchors,{3,3},"Stride"->1,"PaddingSize"->1],
   "reshape1"->ReshapeLayer[{anchors,80,size,size}],"sig1"->LogisticSigmoid,
   "BoxEncodingPredictor"->ConvolutionLayer[4*anchors,{3,3},"Stride"->1,"PaddingSize"->1],
   "reshape2"->ReshapeLayer[{anchors,4,size,size}]},{
   "ClassPredictor"->"reshape1"->"sig1"->NetPort["ClassProb"<>layer],"BoxEncodingPredictor"->"reshape2"->NetPort["Locs"<>layer]}];


ssdConvNet = NetGraph[<|
   "FeatureExtractor/MobilenetV2"->ssdMobileNetFeatureExtractor,
   "BoxPredictor_0"->multiBoxLayer[ 3, "0", 19 ],
   "BoxPredictor_1"->multiBoxLayer[ 6, "1", 10 ],
   "BoxPredictor_2"->multiBoxLayer[ 6, "2", 5 ],
   "BoxPredictor_3"->multiBoxLayer[ 6, "3", 3 ],
   "BoxPredictor_4"->multiBoxLayer[ 6, "4", 2 ],
   "BoxPredictor_5"->multiBoxLayer[ 6, "5", 1 ]|>,{
   NetPort[{"FeatureExtractor/MobilenetV2","multiBox0"}]->"BoxPredictor_0",
   NetPort[{"FeatureExtractor/MobilenetV2","multiBox1"}]->"BoxPredictor_1",
   NetPort[{"FeatureExtractor/MobilenetV2","multiBox2"}]->"BoxPredictor_2",
   NetPort[{"FeatureExtractor/MobilenetV2","multiBox3"}]->"BoxPredictor_3",
   NetPort[{"FeatureExtractor/MobilenetV2","multiBox4"}]->"BoxPredictor_4",
   NetPort[{"FeatureExtractor/MobilenetV2","multiBox5"}]->"BoxPredictor_5"},
   "Input"->NetEncoder[{"Image",{300,300},"ColorSpace"->"RGB"}]];


(* This can reshape both classProbs and regression boxes *)
shuffleLayer[ numberOfAnchors_, height_, width_ ] :=
   NetChain[{ TransposeLayer[{2->4,1->3}], FlattenLayer[2] }];


ssdFlatNet = NetGraph[{
   ssdConvNet,
   shuffleLayer[ 3, 19, 19 ],
   shuffleLayer[ 6, 10, 10 ],
   shuffleLayer[ 6, 5, 5 ],
   shuffleLayer[ 6, 3, 3 ],
   shuffleLayer[ 6, 2, 2 ],
   shuffleLayer[ 6, 1, 1 ],
   CatenateLayer[],
   shuffleLayer[ 3, 19, 19 ],
   shuffleLayer[ 6, 10, 10 ],
   shuffleLayer[ 6, 5, 5 ],
   shuffleLayer[ 6, 3, 3 ],
   shuffleLayer[ 6, 2, 2 ],
   shuffleLayer[ 6, 1, 1 ],
   CatenateLayer[]
   },{
   NetPort[1,"ClassProb0"]->2->8,
   NetPort[1,"ClassProb1"]->3->8,
   NetPort[1,"ClassProb2"]->4->8,
   NetPort[1,"ClassProb3"]->5->8,
   NetPort[1,"ClassProb4"]->6->8,
   NetPort[1,"ClassProb5"]->7->8,
   NetPort[1,"Locs0"]->9->15,
   NetPort[1,"Locs1"]->10->15,
   NetPort[1,"Locs2"]->11->15,
   NetPort[1,"Locs3"]->12->15,
   NetPort[1,"Locs4"]->13->15,
   NetPort[1,"Locs5"]->14->15,
   8->NetPort["ClassProb"],
   15->NetPort["Locs"]
}];


ssdLocsToBoxesNet = NetGraph[{
   "cy"->{PartLayer[{All,1}],ElementwiseLayer[#/10&],ConstantTimesLayer[],ConstantPlusLayer[],ElementwiseLayer[(1-#)*300&]},
   "cx"->{PartLayer[{All,2}],ElementwiseLayer[#/10&],ConstantTimesLayer[],ConstantPlusLayer[],ElementwiseLayer[#*300&]},
   "height"->{PartLayer[{All,3}],ElementwiseLayer[Exp[#*0.2]&],ConstantTimesLayer[],ElementwiseLayer[#*300.&]},
   "width"->{PartLayer[{All,4}],ElementwiseLayer[Exp[#*0.2]&],ConstantTimesLayer[],ElementwiseLayer[#*300.&]}, 
   "minx"->ThreadingLayer[#1-#2/2&],
   "miny"->ThreadingLayer[#1-#2/2&],
   "maxx"->ThreadingLayer[#1+#2/2&],
   "maxy"->ThreadingLayer[#1+#2/2&],
   "cat"->CatenateLayer[],"reshape"->ReshapeLayer[ {4, 1917} ], "transpose"->TransposeLayer[], "reshapePoint"->ReshapeLayer[ { 1917, 2, 2 } ] },{
   {"cx","width"}->"minx",{"cx","width"}->"maxx",
   {"cy","height"}->"miny",{"cy","height"}->"maxy",   
   {"minx","miny","maxx","maxy"}->"cat"->"reshape"->"transpose"->"reshapePoint"->NetPort["Boxes"]
   }];


ssdNet = NetGraph[{
   ElementwiseLayer[(#-.5)*2.0&],
   ssdFlatNet,
   ssdLocsToBoxesNet},{
   1->2,NetPort[{2,"ClassProb"}]->NetPort["ClassProb"],NetPort[{2,"Locs"}]->NetPort[{3,"Input"}],NetPort[{3,"Boxes"}]->NetPort["Boxes"]},
   "Input"->NetEncoder[{"Image",{300,300},"ColorSpace"->"RGB"}]];

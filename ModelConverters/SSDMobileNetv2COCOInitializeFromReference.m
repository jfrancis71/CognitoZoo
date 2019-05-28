(* ::Package:: *)

paramFileName = "/home/julian/SSDMobileNet/SSDMobileNet.hdf";


initConvRules[ depthwise_, includeBiases_ ][ HDFPrefix_ ] := Module[{
   weights = If[depthwise,Transpose[Import[paramFileName,HDFPrefix<>"/depthwise_weights"],{3,4,1,2}],Transpose[Import[paramFileName,HDFPrefix<>"/weights"],{3,4,2,1}]]},
   Module[{
   biases = If[includeBiases,Import[paramFileName,HDFPrefix<>"/biases"],ConstantArray[0,Length@weights]]},
      {{"Weights"}->weights,{"Biases"}->biases} ] ]


initBatchNormalizationRules[ HDFPrefix_ ] := Module[{
   batchNormParams = Import[paramFileName,{"Datasets",{HDFPrefix<>"/BatchNorm/gamma",
      HDFPrefix<>"/BatchNorm/beta",HDFPrefix<>"/BatchNorm/moving_mean",HDFPrefix<>"/BatchNorm/moving_variance"}}]},
   {{"Epsilon"}->0.0010000000474974513, {"Scaling"}->batchNormParams[[1]],
      {"Biases"}->batchNormParams[[2]], {"MovingMean"}->batchNormParams[[3]], {"MovingVariance"}->batchNormParams[[4]] } ]


(* rules should be a list of net replacement rules, prefix will be prepended to the lhs of each rule *)
prefixRules[ rules_, prefix_ ] :=
   Prepend[ #[[1]],prefix]->#[[2]]&/@rules


initRules[ initComponents_ ][ HDFPrefix_ ] :=
   Flatten@Map[ Function[ layer, prefixRules[ layer[[2]][ HDFPrefix<>"/"<>layer[[1]] ], layer[[1]] ] ], initComponents ]


initMobileUnitRules[ depthwise_ ][ HDFPrefix_ ] := Join[
   prefixRules[ initConvRules[ depthwise, False ][ HDFPrefix ], "conv2d" ],
   prefixRules[ initBatchNormalizationRules[ HDFPrefix ], "BatchNorm" ]
];


initMobileNetBlockRules[ HDFPrefix_ ] := initRules[ {
   "expand"->initMobileUnitRules[ False ],
   "depthwise"->initMobileUnitRules[ True ],
   "project"->initMobileUnitRules[ False ] } ][ HDFPrefix ];


initInvResMobileNetBlockRules[ HDFPrefix_ ] := prefixRules[ initMobileNetBlockRules[ HDFPrefix ], 1 ]


initFeatureNetRules[ HDFPrefix_ ] := initRules[ {
   "Conv"->initMobileUnitRules[ False ],
   "expanded_conv/depthwise"->initMobileUnitRules[ True ],
   "expanded_conv/project"->initMobileUnitRules[ False ],
   "expanded_conv_1"->initMobileNetBlockRules,
   "expanded_conv_2"->initInvResMobileNetBlockRules,
   "expanded_conv_3"->initMobileNetBlockRules,
   "expanded_conv_4"->initInvResMobileNetBlockRules,
   "expanded_conv_5"->initInvResMobileNetBlockRules,
   "expanded_conv_6"->initMobileNetBlockRules,
   "expanded_conv_7"->initInvResMobileNetBlockRules,
   "expanded_conv_8"->initInvResMobileNetBlockRules,
   "expanded_conv_9"->initInvResMobileNetBlockRules,
   "expanded_conv_10"->initMobileNetBlockRules,
   "expanded_conv_11"->initInvResMobileNetBlockRules,
   "expanded_conv_12"->initInvResMobileNetBlockRules,
   "expanded_conv_13/expand"->initMobileUnitRules[ False ],
   "expanded_conv_13/depthwise"->initMobileUnitRules[ True ],
   "expanded_conv_13/project"->initMobileUnitRules[ False ],
   "expanded_conv_14"->initInvResMobileNetBlockRules,
   "expanded_conv_15"->initInvResMobileNetBlockRules,
   "expanded_conv_16"->initMobileNetBlockRules,
   "Conv_1"->initMobileUnitRules[ False ],   
   "layer_19_1_Conv2d_2_1x1_256"->initMobileUnitRules[ False ],
   "layer_19_2_Conv2d_2_3x3_s2_512_depthwise"->initMobileUnitRules[ True ],
   "layer_19_2_Conv2d_2_3x3_s2_512"->initMobileUnitRules[ False ],
   "layer_19_1_Conv2d_3_1x1_128"->initMobileUnitRules[ False ],
   "layer_19_2_Conv2d_3_3x3_s2_256_depthwise"->initMobileUnitRules[ True ],
   "layer_19_2_Conv2d_3_3x3_s2_256"->initMobileUnitRules[ False ],
   "layer_19_1_Conv2d_4_1x1_128"->initMobileUnitRules[ False ],   
   "layer_19_2_Conv2d_4_3x3_s2_256_depthwise"->initMobileUnitRules[ True ],   
   "layer_19_2_Conv2d_4_3x3_s2_256"->initMobileUnitRules[ False ],   
   "layer_19_1_Conv2d_5_1x1_64"->initMobileUnitRules[ False ],
   "layer_19_2_Conv2d_5_3x3_s2_128_depthwise"->initMobileUnitRules[ True ],   
   "layer_19_2_Conv2d_5_3x3_s2_128"->initMobileUnitRules[ False ]
} ][ HDFPrefix ];


nonCOCOClasses = { {1}, {13}, {27}, {30}, {31}, {46}, {67}, {69}, {70}, {72}, {84} }; (* including the first one, the no object class *)


initMultiBoxRules[ anchorBoxes_ ][ HDFPrefix_ ] := initRules[ {
   "ClassPredictor"->Function[ HDFClassPrefix, {
      {"Weights"}->Delete[Transpose[Import[paramFileName,HDFClassPrefix<>"/weights"],{3,4,2,1}],
      Flatten[Table[nonCOCOClasses+p*91, {p,0,anchorBoxes-1}],1] ],
      {"Biases"}->Delete[Import[paramFileName,HDFClassPrefix<>"/biases"],
      Flatten[Table[nonCOCOClasses+p*91, {p,0,anchorBoxes-1}],1] ]      
       } ],
   "BoxEncodingPredictor"->initConvRules[ False, True ]
   }
][ HDFPrefix ];


initConvNetRules = initRules[ {
   "FeatureExtractor/MobilenetV2"->initFeatureNetRules,
   "BoxPredictor_0"->initMultiBoxRules[3],
   "BoxPredictor_1"->initMultiBoxRules[6],
   "BoxPredictor_2"->initMultiBoxRules[6],
   "BoxPredictor_3"->initMultiBoxRules[6],
   "BoxPredictor_4"->initMultiBoxRules[6],
   "BoxPredictor_5"->initMultiBoxRules[6]
} ][ "" ];


initFlatNetRules = prefixRules[ initConvNetRules, 1 ];


initConstantTimesRules[ HDFPrefix_ ] := { {"Scaling"}->Import[paramFileName, HDFPrefix ] }


initConstantPlusRules[ HDFPrefix_ ] := { {"Biases"}->Import[paramFileName, HDFPrefix ] }


initLocsToBoxesNetRules[ HDFPrefix_ ] := Join[
   Fold[ prefixRules, initConstantTimesRules[ HDFPrefix<>"/sub_1" ], { 3, "height" } ],
   Fold[ prefixRules, initConstantTimesRules[ HDFPrefix<>"/sub"], { 3, "width" } ],
   Fold[ prefixRules, initConstantTimesRules[ HDFPrefix<>"/sub_1"], { 3, "cy" } ],
   Fold[ prefixRules, initConstantTimesRules[ HDFPrefix<>"/sub"], { 3, "cx" } ],
   Fold[ prefixRules, initConstantPlusRules[ HDFPrefix<>"/add"], { 4, "cy" } ],
   Fold[ prefixRules, initConstantPlusRules[ HDFPrefix<>"/add_1"], { 4, "cx" } ]
];


initSSDNetRules = Join[
   prefixRules[ initFlatNetRules, 2 ],
   prefixRules[ initLocsToBoxesNetRules["Postprocessor/Decode/get_center_coordinates_and_sizes"], 3 ]
];


initSSDNet = NetReplacePart[ssdNet, initSSDNetRules ];

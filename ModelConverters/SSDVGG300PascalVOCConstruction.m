(* ::Package:: *)

conv1[n_?IntegerQ]:=Sequence[
   ConvolutionLayer[n,1,"PaddingSize"->0,"Stride"->1],
   Ramp
];
conv3[n_?IntegerQ]:=Sequence[
   ConvolutionLayer[n,3,"PaddingSize"->1,"Stride"->1],
   Ramp
];


blockNet4 = NetChain[{
   "conv1a"->{conv3[64]},
   "conv1b"->{conv3[64]},
   "pool1"->PoolingLayer[{2,2},"Stride"->2],
   "conv2a"->{conv3[128]},
   "conv2b"->{conv3[128]},
   "pool2"->PoolingLayer[{2,2},"Stride"->2],
   "conv3a"->{conv3[256]},
   "conv3b"->{conv3[256]},
   "conv3c"->{conv3[256]},
   "pool3"->{PaddingLayer[{{0,0},{0,1},{0,1}}],PoolingLayer[{2,2},"Stride"->2]},
   "conv4a"->{conv3[512]},
   "conv4b"->{conv3[512]},
   "conv4c"->{conv3[512]}
   }];


blockNet7 = NetChain[{
  "pool4"->PoolingLayer[{2,2},"Stride"->2],
   "conv5a"->{conv3[512]},
   "conv5b"->{conv3[512]},
   "conv5c"->{conv3[512]},
   "pool5"->PoolingLayer[{3,3},"Stride"->1,"PaddingSize"->1],
   "conv6"->{ConvolutionLayer[1024,{3,3},"Dilation"->6,"PaddingSize"->6],Ramp},
   "conv7"->{conv1[1024]}
}];


blockNet8 = NetChain[{
   "conv8a"->{conv1[256]},
   "pad2"->PaddingLayer[{{0,0},{1,1},{1,1}}],
   "conv8b"->{ConvolutionLayer[512, {3,3}, "Stride"->2],Ramp}
}];


blockNet9 = NetChain[{
   "conv9a"->{conv1[128]},
   "conv9b"->{ConvolutionLayer[256, {3,3},"Stride"->2,"PaddingSize"->1],Ramp}
}];


blockNet10 = NetChain[{
   "conv10a"->{conv1[128]},
   "conv10b"->{ConvolutionLayer[256, {3,3}],Ramp}
}];


blockNet11 = NetChain[{
   "conv11a"->{conv1[128]},
   "conv11b"->{ConvolutionLayer[256, {3,3}],Ramp}
}];


multiBoxLocationDecoder[ numberOfAnchors_, width_, height_ ] :=
   NetGraph[{
      "convloc"->{ConvolutionLayer[ numberOfAnchors*4, {3,3}, "PaddingSize"->1 ]},
      "reshape"->ReshapeLayer[{numberOfAnchors, 4, height, width }],
      "cx"->{PartLayer[{All,1}],ConstantTimesLayer[],ConstantPlusLayer[]},
      "cy"->{PartLayer[{All,2}],ConstantTimesLayer[],ConstantPlusLayer[]},
      "width"->{PartLayer[{All,3}],ElementwiseLayer[Exp[#*0.2]&],ConstantTimesLayer[]},
      "height"->{PartLayer[{All,4}],ElementwiseLayer[Exp[#*0.2]&],ConstantTimesLayer[]},
      "catenate"->CatenateLayer[],
      "locs"->ReshapeLayer[{4,numberOfAnchors,height,width}]},{
      
      "convloc"->"reshape"->{"cx","cy","width","height"}->"catenate"->"locs"}]


multiBoxClassesDecoder[ numberOfAnchors_, width_, height_ ] :=
   NetChain[{ ConvolutionLayer[ numberOfAnchors*21, {3,3}, "PaddingSize"->1 ], ReshapeLayer[{numberOfAnchors,21,height,width}],TransposeLayer[2->4],TransposeLayer[2->3],SoftmaxLayer[] }];


channelNormalizationNet = NetChain[{
   TransposeLayer[1->3],
   ElementwiseLayer[Log[Max[#^2,10^-6]]&],SoftmaxLayer[],
   ElementwiseLayer[Sqrt],TransposeLayer[1->3],
   ConstantTimesLayer[]
}];


multiBoxLayer1 = NetGraph[{
   "channelNorm1"->channelNormalizationNet,
   "multiboxClasses"->multiBoxClassesDecoder[ 4, 38, 38 ],
   "multiboxLocs"->multiBoxLocationDecoder[ 4, 38, 38 ]},
   {"channelNorm1"->{"multiboxClasses"->NetPort["ClassProb1"],"multiboxLocs"->NetPort["Boxes1"]}}];


multiBoxLayer2 = NetGraph[{
   "multiboxClasses"->multiBoxClassesDecoder[ 6, 19, 19 ],
   "multiboxLocs"->multiBoxLocationDecoder[ 6, 19, 19 ]},
   {"multiboxClasses"->NetPort["ClassProb2"],"multiboxLocs"->NetPort["Boxes2"]}];


multiBoxLayer3 = NetGraph[{
   "multiboxClasses"->multiBoxClassesDecoder[ 6, 10, 10 ],
   "multiboxLocs"->multiBoxLocationDecoder[ 6, 10, 10 ]},
   {"multiboxClasses"->NetPort["ClassProb3"],"multiboxLocs"->NetPort["Boxes3"]}];


multiBoxLayer4 = NetGraph[{
   "multiboxClasses"->multiBoxClassesDecoder[ 6, 5, 5 ],
   "multiboxLocs"->multiBoxLocationDecoder[ 6, 5, 5 ]},
   {"multiboxClasses"->NetPort["ClassProb4"],"multiboxLocs"->NetPort["Boxes4"]}];


multiBoxLayer5 = NetGraph[{
   "multiboxClasses"->multiBoxClassesDecoder[ 4, 3, 3 ],
   "multiboxLocs"->multiBoxLocationDecoder[ 4, 3, 3 ]},
   {"multiboxClasses"->NetPort["ClassProb5"],"multiboxLocs"->NetPort["Boxes5"]}];


multiBoxLayer6 = NetGraph[{
   "multiboxClasses"->multiBoxClassesDecoder[ 4, 1, 1 ],
   "multiboxLocs"->multiBoxLocationDecoder[ 4, 1, 1 ]},
   {"multiboxClasses"->NetPort["ClassProb6"],"multiboxLocs"->NetPort["Boxes6"]}];


ssdNet = NetGraph[{
   "blockNet4"->blockNet4,
   "blockNet7"->blockNet7,
   "blockNet8"->blockNet8,
   "blockNet9"->blockNet9,
   "blockNet10"->blockNet10,
   "blockNet11"->blockNet11,
   "multiboxLayer1"->multiBoxLayer1,
   "multiboxLayer2"->multiBoxLayer2,
   "multiboxLayer3"->multiBoxLayer3,
   "multiboxLayer4"->multiBoxLayer4,
   "multiboxLayer5"->multiBoxLayer5,
   "multiboxLayer6"->multiBoxLayer6
   },
   {"blockNet4"->"blockNet7"->"blockNet8"->"blockNet9"->"blockNet10"->"blockNet11",
   "blockNet4"->"multiboxLayer1",
   "blockNet7"->"multiboxLayer2",
   "blockNet8"->"multiboxLayer3",
   "blockNet9"->"multiboxLayer4",
   "blockNet10"->"multiboxLayer5",
   "blockNet11"->"multiboxLayer6"
   },
   "Input"->NetEncoder[{"Image",{300,300},"ColorSpace"->"RGB","MeanImage"->{123,117,104}/255.}]];

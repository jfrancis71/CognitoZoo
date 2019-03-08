(* ::Package:: *)

hdfFile = "/Users/julian/yolov3/darknet/Yolov3OpenImages.h5";


YoloConvLayerInitRules[ netName_, hdfName_ ] := {
   {netName,1,"Weights"}->Import[hdfFile,{"Datasets",hdfName<>"_weights"}],
   {netName,2,"Beta"}->Import[hdfFile,{"Datasets",hdfName<>"_biases"}],
   {netName,2,"MovingMean"}-> Import[hdfFile,{"Datasets",hdfName<>"_rolling_mean"}],
   {netName,2,"MovingVariance"}->Import[hdfFile,{"Datasets",hdfName<>"_rolling_variance"}],
   {netName,2,"Gamma"}-> Import[hdfFile,{"Datasets",hdfName<>"_scales"}]
};


SmallResidualBlockInitRules[ name_String, firstLayer_ ] := Map[Prepend[#[[1]],name]->#[[2]]&,Join[
   YoloConvLayerInitRules[ 1, "layer"<>ToString[ firstLayer ] ],
   YoloConvLayerInitRules[ 2, "layer"<>ToString[ firstLayer +1]]
]];


LargeResidualBlockInitRules[ name_String, firstLayer_Integer ] := Map[Prepend[#[[1]],name]->#[[2]]&,Join[
   YoloConvLayerInitRules[ 1, "layer"<>ToString[firstLayer] ],
   YoloConvLayerInitRules[ 2, "layer"<>ToString[firstLayer+1] ],
   YoloConvLayerInitRules[ 3, "layer"<>ToString[firstLayer+2] ]
]];


LargeResidualBlockInitRules[ "layer4", 1 ]


YoloConvLayerInitRules[ "layer0"]


yoloConvNetInitRules = Join[
   YoloConvLayerInitRules[ "layer0", "layer0"],
   LargeResidualBlockInitRules[ "layer4", 1 ],
   LargeResidualBlockInitRules[ "layer8", 5 ],
   SmallResidualBlockInitRules[ "layer11", 9 ],
   LargeResidualBlockInitRules[ "layer15", 12 ],
   SmallResidualBlockInitRules[ "layer18", 16 ],
   SmallResidualBlockInitRules[ "layer21", 19 ],
   SmallResidualBlockInitRules[ "layer24", 22 ],
   SmallResidualBlockInitRules[ "layer27", 25 ],
   SmallResidualBlockInitRules[ "layer30", 28 ],
   SmallResidualBlockInitRules[ "layer33", 31 ],
   SmallResidualBlockInitRules[ "layer36", 34 ],
   LargeResidualBlockInitRules[ "layer40", 37 ],
   SmallResidualBlockInitRules[ "layer43", 41 ],
   SmallResidualBlockInitRules[ "layer46", 44 ],
   SmallResidualBlockInitRules[ "layer49", 47 ],
   SmallResidualBlockInitRules[ "layer52", 50 ],
   SmallResidualBlockInitRules[ "layer55", 53 ],
   SmallResidualBlockInitRules[ "layer58", 56 ],
   SmallResidualBlockInitRules[ "layer61", 59 ],
   LargeResidualBlockInitRules[ "layer65", 62 ],
   SmallResidualBlockInitRules[ "layer68", 66 ],
   SmallResidualBlockInitRules[ "layer71", 69 ],
   SmallResidualBlockInitRules[ "layer74", 72 ],
   YoloConvLayerInitRules[ "layer75", "layer75"],
   YoloConvLayerInitRules[ "layer76", "layer76"],
   YoloConvLayerInitRules[ "layer77", "layer77"],
   YoloConvLayerInitRules[ "layer78", "layer78"],
   YoloConvLayerInitRules[ "layer79", "layer79"],
   YoloConvLayerInitRules[ "layer80", "layer80"],
   {{"layer81","Weights"}->Import[hdfFile,{"Datasets","layer81_weights"}], {"layer81","Biases"}->Import[hdfFile,{"Datasets","layer81_biases"}]},
   YoloConvLayerInitRules[ "layer84", "layer84"],
   {{"layer85","Weights"}->Table[If[j==i,1,0],{j,1,256},{i,1,256},{2},{2}], {"layer85","Biases"}->ConstantArray[0,256]},
   YoloConvLayerInitRules[ "layer87", "layer87"],
   YoloConvLayerInitRules[ "layer88", "layer88"],
   YoloConvLayerInitRules[ "layer89", "layer89"],
   YoloConvLayerInitRules[ "layer90", "layer90"],
   YoloConvLayerInitRules[ "layer91", "layer91"],
   YoloConvLayerInitRules[ "layer92", "layer92"],
   {{"layer93","Weights"}->Import[hdfFile,{"Datasets","layer93_weights"}], {"layer93","Biases"}->Import[hdfFile,{"Datasets","layer93_biases"}]},
   YoloConvLayerInitRules[ "layer96", "layer96"],
   {{"layer97","Weights"}->Table[If[j==i,1,0],{j,1,128},{i,1,128},{2},{2}], {"layer97","Biases"}->ConstantArray[0,128]},
   YoloConvLayerInitRules[ "layer99", "layer99"],
   YoloConvLayerInitRules[ "layer100", "layer100"],
   YoloConvLayerInitRules[ "layer101", "layer101"],
   YoloConvLayerInitRules[ "layer102", "layer102"],
   YoloConvLayerInitRules[ "layer103", "layer103"],
   YoloConvLayerInitRules[ "layer104", "layer104"],
   {{"layer105","Weights"}->Import[hdfFile,{"Datasets","layer105_weights"}], {"layer105","Biases"}->Import[hdfFile,{"Datasets","layer105_biases"}]}
];


widthScales = {
   Table[{116,156,373}[[n]],{n,1,3},{19},{19}],
   Table[{30,62,59}[[n]],{n,1,3},{38},{38}],
   Table[{10,16,33}[[n]],{n,1,3},{76},{76}]};


heightScales = {
   Table[{90,198,326}[[n]],{n,1,3},{19},{19}],
   Table[{61,45,119}[[n]],{n,1,3},{38},{38}],
   Table[{13,30,23}[[n]],{n,1,3},{76},{76}]};


LocationInitRules[ name_, layorNo_, anchors_, width_, height_ ] := {
   { name, "height", 3, "Scaling" } -> heightScales[[layorNo]],
   { name, "width", 3, "Scaling" } -> widthScales[[layorNo]],
   { name, "cx", 3, "Biases" } -> Table[j,{anchors},{i,0,height-1},{j,0,width-1}],
   { name, "cy", 3, "Biases" } -> Table[i,{anchors},{i,0,height-1},{j,0,width-1}]
};


yoloDecoderNetInitRules = Join[
   LocationInitRules[ "locationsmap1", 1, 3, 19, 19 ],
   LocationInitRules[ "locationsmap2", 2, 3, 38, 38 ],
   LocationInitRules[ "locationsmap3", 3, 3, 76, 76 ]
];


yoloOpenImageNetRules = Join[
   Map[ Prepend[#[[1]],1]->#[[2]]&, yoloConvNetInitRules ],
   Map[ Prepend[#[[1]],2]->#[[2]]&, yoloDecoderNetInitRules ]
];


yoloOpenImagesNetInit = NetReplacePart[ yoloOpenImagesNet, yoloOpenImageNetRules ];

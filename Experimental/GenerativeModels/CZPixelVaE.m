(* ::Package:: *)

<<"Experimental/GenerativeModels/CZVariationalAutoencoders.m"
<<"Experimental/GenerativeModels/CZPixelCNN.m"


CZLatentModelQ[ CZPixelVaE[ _ ] ] := True;


CZPixelVaEDecoder[ crossEntropyType_, imageDims_ ] := NetGraph[{
   {500,Ramp},
   {500,Ramp},
   {imageDims[[1]]*imageDims[[2]]},
   ReshapeLayer[imageDims],
   CZCreatePixelCNNConditionalNet[ crossEntropyType, PixelCNNOrdering[ imageDims ] ]},{
   NetPort["Conditional"]->1->2->3->4->NetPort[{5,"Conditional"}],
   NetPort["Target"]->NetPort[{5,"Input"}]}];


CZCreatePixelVaE[ type_:CZBinary[{28,28}], latentUnits_:8 ] := CZGenerativeModel[
   CZPixelVaE[ latentUnits ], 
   type,
   CZEncoder[ type ],
   CZCreateVaENet[ CZCreateEncoder[ latentUnits ], CZPixelVaEDecoder[ type, type[[1]] ] ]];


SyntaxInformation[ CZPixelVaE ]= {"ArgumentsPattern"->{_}};


CZSample[ CZGenerativeModel[ CZPixelVaE[ latentUnits_ ], inputType_, encoder_, pixelCNNNet_ ] ] := (
   z = CZSampleVaELatent[ latentUnits ];
   cond = NetTake[ NetExtract[ pixelCNNNet, "decoder" ],{1,4} ][ z ];
   CZSampleConditionalPixelCNN[ NetExtract[ pixelCNNNet, {"decoder",5} ], inputType, encoder, cond ]
)


CZModelLRM[ CZPixelVaE[ _ ] ] := Flatten[Table[
   {{"decoder",5,"predict"<>ToString[k],"masked_input"}->0,
   {"decoder",5,"loss"<>ToString[k],"mask"}->0},{k,4}],1]

(* ::Package:: *)

<<"Experimental/GenerativeModels/CZVariationalAutoencoders.m"
<<"Experimental/GenerativeModels/CZPixelCNN.m"


CZLatentModelQ[ CZPixelVaE[ _ ] ] := True;


CZPixelVaEDecoder[ inputType_ ] := NetGraph[{
   {500,Ramp},
   {500,Ramp},
   {inputType[[1,1]]*inputType[[1,2]]},
   ReshapeLayer[inputType[[1,1;;2]]],
   CZCreatePixelCNNConditionalNet[ inputType, PixelCNNOrdering[ inputType[[1]] ] ]},{
   NetPort["Conditional"]->1->2->3->4->NetPort[{5,"Conditional"}],
   NetPort["Target"]->NetPort[{5,"Input"}]}];


CZCreatePixelVaE[ type_:CZBinary[{28,28}], latentUnits_:8 ] := CZGenerativeModel[
   CZPixelVaE[ latentUnits ], 
   type,
   CZCreateVaENet[ CZCreateEncoder[ latentUnits ], CZPixelVaEDecoder[ type ] ]];


SyntaxInformation[ CZPixelVaE ]= {"ArgumentsPattern"->{_}};


CZSample[ CZGenerativeModel[ CZPixelVaE[ latentUnits_ ], inputType_, pixelCNNNet_ ] ] := (
   z = CZSampleVaELatent[ latentUnits ];
   cond = NetTake[ NetExtract[ pixelCNNNet, "decoder" ],{1,4} ][ z ];
   CZSampleConditionalPixelCNN[ NetExtract[ pixelCNNNet, {"decoder",5} ], inputType, cond ]
)


CZModelLRM[ CZPixelVaE[ _ ] ] := Flatten[Table[
   {{"decoder",5,"predict"<>ToString[k],"masked_input"}->0,
   {"decoder",5,"loss"<>ToString[k],"mask"}->0},{k,4}],1]

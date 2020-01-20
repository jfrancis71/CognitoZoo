(* ::Package:: *)

(*
   PixelVAE: A Latent Variable Model For Natural Images, 2016
   Gulrajani, Kumar, Ahmed, Ali Taiga, Visin, Vazquez, Courville
   
   Note, this codes implements the above idea of marrying the
   Variation Autoencoder with PixelCNN, but does not implement
   the hierarchical aspect.
*)


<<"Experimental/GenerativeModels/CZVariationalAutoencoders.m"
<<"Experimental/GenerativeModels/CZPixelCNN.m"


CZLatentModelQ[ CZPixelVaE[ _ ] ] := True;


CZPixelVaEDecoder[ inputType_ ] := NetGraph[{
   {500,Ramp},
   {500,Ramp},
   {Apply[Times,inputType[[1,-2;;-1]]]},
   ReshapeLayer[inputType[[1,-2;;-1]]],
   CZCreatePixelCNNConditionalNet[ inputType, PixelCNNOrder[ inputType[[1]] ] ]},{
   NetPort["Conditional"]->1->2->3->4->NetPort[{5,"Conditional"}],
   NetPort["Target"]->NetPort[{5,"Input"}]}];


CZCreatePixelVaE[ type_:CZBinary[{28,28}], latentUnits_:8 ] := CZGenerativeModel[
   CZPixelVaE[ latentUnits ], 
   type,
   CZCreateVaENet[ CZCreateEncoder[ latentUnits ], CZPixelVaEDecoder[ type ] ]];


SyntaxInformation[ CZPixelVaE ]= {"ArgumentsPattern"->{_}};


CZSample[ CZGenerativeModel[ CZPixelVaE[ latentUnits_ ], inputType_, pixelCNNNet_ ] ] := (
   z = CZSampleStandardNormalDistribution[ {latentUnits} ];
   cond = NetTake[ NetExtract[ pixelCNNNet, "decoder" ],{1,4} ][ z ];
   CZSampleConditionalPixelCNN[ NetExtract[ pixelCNNNet, {"decoder",5} ], inputType, cond ]
)


CZModelLRM[ CZPixelVaE[ _ ], dims_ ] := Flatten[Table[
   {{"decoder",5,"predict"<>ToString[k],"masked_input"}->0,
   {"decoder",5,"loss"<>ToString[k],"mask"}->0},{k,If[Length[dims]==2,4,4*dims[[1]]]}],1]

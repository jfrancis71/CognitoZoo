(* ::Package:: *)

(*
wt is a weight vector
*)
CZBayesProbT[ t_, wt_ ] := Exp[wt[[t]]]/Total[Exp[wt]];


(*
   pcond[v,t,w] v is a vector, t is a scalar between 1...N, w is a matrix with N rows and |v| columns
 returns a scalar
*)
CZBayesCondProbVonT[ v_, t_, w_ ] := Apply[Times, LogisticSigmoid[w[[t]]]*v+(1-LogisticSigmoid[w[[t]]])*(1-v)];


(*
   jt[v,t,{w,wt}] is the joint distribution for (v,t) where w is conditional prob matrix and wt is vector prob matrix
*)
CZBayesJointT[ v_, t_, {w_,wt_}] := CZBayesCondProbVonT[v,t,w]*CZBayesProbT[t,wt];


(*
   v[v,w,nt] is marginal distribution over v given structure w which is {w,wt}
*)
CZBayesMarginalV[ v_, w_, nt_] := Sum[CZBayesJointT[v,t,w],{t,1,nt}];


(*
   l[v,w,nt] is loss nt is number of entries in t.
*)
CZBayesLoss[ v_, w_, nt_] := Log[CZBayesMarginalV[v,w,nt]];


(*
   dat=Join[Tuples[{0,1},2],ConstantArray[{1,1},8]];
   BayesMLE[dat,2]
   returns {-10.043858601430028`,{w[1,1]\[Rule]-0.05589845026206474`,w[1,2]\[Rule]-0.0548370752934956`,w[2,1]\[Rule]5.016771454675282`,w[2,2]\[Rule]4.998251823564115`,wt[1]\[Rule]0.5286574893407507`,wt[2]\[Rule]1.3031577035237136`}}
*)


(*
   data is in Examples*Features format, N is number of hidden states
*)
CZBayesMLE[data_,N_] := Module[{fw,fwt,loss,tw1},
   fw=Table[w[t,k],{t,1,N},{k,1,Length[data[[1]]]}];
   fwt=Table[wt[t],{t,1,N}];
   loss=Total[CZBayesLoss[#,{fw,fwt},N]&/@data];
   tw1=Join[
      Flatten[Table[{w[t,k],Random[]},{t,1,N},{k,1,Length[data[[1]]]}],1],
      Table[{wt[t],Random[]},{t,1,N}]
   ];
   FindMaximum[loss,tw1]
];

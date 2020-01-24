(* ::Package:: *)

<<"Experimental/GenerativeModels/CZNBModels.m"


mnist=ResourceData["MNIST","TrainingData"][[;;500,1]];


binImages = {ImageData[#]}&/@Binarize/@mnist;


discImages = CZDiscretize/@mnist;


images={ImageData[#]}&/@mnist;


nbmodel1 = CZCreateNBModel[];


nbtrain1 = CZTrain[ nbmodel1, binImages, MaxTrainingRounds->1  ];


CZLogDensity[ nbtrain1, binImages[[1]] ]


Image@First@CZSample[ nbtrain1 ]


nbmodel2 = CZCreateNBModel[ CZDiscrete[{1,28,28}] ];


nbtrain2 = CZTrain[ nbmodel2, discImages, MaxTrainingRounds->1  ];


CZLogDensity[ nbtrain2, discImages[[1]] ]


Image@First@CZSample[ nbtrain2 ]


nbmodel3 = CZCreateNBModel[ CZRealGauss[{1,28,28}] ];


nbtrain3 = CZTrain[ nbmodel3, images, MaxTrainingRounds->1  ];


CZLogDensity[ nbtrain3, images[[1]] ]


Image@First@CZSample[ nbtrain3 ]


<<"Experimental/GenerativeModels/CZVariationalAutoencoders.m"


vaemodel1 = CZCreateVaE[];


vaetrain1 = CZTrain[ vaemodel1, binImages, MaxTrainingRounds->1  ];


CZLogDensity[ vaetrain1, binImages[[1]] ]


Image@First@CZSample[ vaetrain1 ]


vaemodel2 = CZCreateVaE[ CZDiscrete[{1,28,28}] ];


vaetrain2 = CZTrain[ vaemodel2, discImages, MaxTrainingRounds->1  ];


CZLogDensity[ vaetrain2, discImages[[1]] ]


Image@First@CZSample[ vaetrain2 ]


vaemodel3 = CZCreateVaE[ CZRealGauss[{1,28,28}] ];


vaetrain3 = CZTrain[ vaemodel3, images, MaxTrainingRounds->1  ];


CZLogDensity[ vaetrain3, images[[1]] ]


Image@First@CZSample[ vaetrain3 ]


<<"Experimental/GenerativeModels/CZPixelCNN.m"


cnnmodel1 = CZCreatePixelCNN[];


cnntrain1 = CZTrain[ cnnmodel1, binImages, MaxTrainingRounds->1  ];


CZLogDensity[ cnntrain1, binImages[[1]] ]


Image@First@CZSample[ cnntrain1 ]


cnnmodel2 = CZCreatePixelCNN[ CZDiscrete[{1,28,28}] ];


cnntrain2 = CZTrain[ cnnmodel2, discImages, MaxTrainingRounds->1  ];


CZLogDensity[ cnntrain2, discImages[[1]] ]


Image@First@CZSample[ cnntrain2 ]


cnnmodel3 = CZCreatePixelCNN[ CZRealGauss[{1,28,28}] ];


cnntrain3 = CZTrain[ cnnmodel3, images, MaxTrainingRounds->1  ];


CZLogDensity[ cnntrain3, images[[1]] ]


Image@First@CZSample[ cnntrain3 ]


<<"Experimental/GenerativeModels/CZPixelVaE.m"


pixelvaemodel1 = CZCreatePixelVaE[];


pixelvaetrain1 = CZTrain[ pixelvaemodel1, binImages, MaxTrainingRounds->1  ];


CZLogDensity[ pixelvaetrain1, binImages[[1]] ]


Image@First@CZSample[ pixelvaetrain1 ]


pixelvaemodel2 = CZCreatePixelVaE[ CZDiscrete[{1,28,28}] ];


pixelvaetrain2 = CZTrain[ pixelvaemodel2, discImages, MaxTrainingRounds->1  ];


CZLogDensity[ pixelvaetrain2, discImages[[1]] ]


Image@First@CZSample[ pixelvaetrain2 ]


pixelvaemodel3 = CZCreatePixelVaE[ CZRealGauss[{1,28,28}] ];


pixelvaetrain3 = CZTrain[ pixelvaemodel3, images, MaxTrainingRounds->1  ];


CZLogDensity[ pixelvaetrain3, images[[1]] ]


Image@First@CZSample[ pixelvaetrain3 ]


<<"Experimental/GenerativeModels/CZRealNVPConv.m"


realnvp = CZCreateRealNVP[{1,32,32}];


multichannel={ImageData[ImageResize[#,{32,32}],Interleaving->False]}&/@mnist;


realnvptrain=CZTrain[ realnvp, multichannel, MaxTrainingRounds->1]


CZLogDensity[ realnvptrain, multichannel[[67]]]


Image@CZSample[ realnvptrain ][[1]]

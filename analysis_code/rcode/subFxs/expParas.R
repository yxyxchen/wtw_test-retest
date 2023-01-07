# experiment design paramters
conditions = c("HP", "LP")
delayMaxs = c(12, 24) # max delay durations in secs
nBlock = 2
blockMin = 10 # block duration in mins
blockSec = blockMin * 60 # block duration in secs
tokenValue = 2 # value of the matured token
iti = 1.5

# save 
save("conditions" = conditions,
     "delayMaxs" = delayMaxs,
     "blockMin" = blockMin,
     "blockSec" = blockSec,
     "nBlock" = nBlock,
     "tokenValue" = tokenValue,
     file = "expParas.RData")


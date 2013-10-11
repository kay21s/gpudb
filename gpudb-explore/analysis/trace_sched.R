###################################################################
#               Query Trace Scheduling Simulation
###################################################################
library(BB)

kq_SQF_sim = function(files, reader=files2dur) {
	qs <- reader(files)
	return(kq_SQF(qs))
}

#############################
#  input: query(s) duration
# output: query(s) slowdown
#############################
kq_SQF = function(qs) {
	qo <- order(qs)
	qs2 <- qs[qo]	

	sl <- cumsum(qs2)/qs2
	sl2 <- sl[order(qo)]
	
	return(sl2)
}


kq_FO_sim = function(files, reader=files2dur) {
	qs <- reader(files)
	return(kq_FO(qs))
}

kq_FO = function(qs) {
	qo <- order(qs)
	qs2 <- qs[qo]	
	sl <- qs	

	stail <- 0
	ltail <- 0
	k <- 1
	for (q in qs2) {
		sl[k] <- (q + stail)/q
		tmp <- ltail
		ltail <- stail + q
		stail <- tmp
		k <- k + 1
	}
	
	return(sl[order(qo)])
}

kqr_MSH_sim = function(files, reader=files2dur) {
	qs <- reader(files)
	return(kqr_MSH(qs))
}

kqr_MSH = function(qs, gen=100) {
	#maintain the order of all scheduled queries
	qnum  <- length(qs)
	log   <- order(qs)
	occur <- rep(1,length(qs))

	#sum((time_passed/could occured)/have_occured)/#query
	delta <- rep(0, length(qs))
	csl <- 0
	maxq <- gen*qnum
	for (i in seq(1, maxq)) {
		sl    <- sum((sum(qs * occur)/qs)/occur) /sum(qnum)
		csl <- csl + sl
		for (j in order(qs)) {
			occur[j] <- occur[j] + 1
			delta[j] <- sum((sum(qs * occur)/qs)/occur) /sum(qnum) - sl
			occur[j] <- occur[j] - 1
		}
		if (i==maxq)
			break

		index <- which(delta == min(delta))[1]
		log <- append(log, index)
		occur[index] <- occur[index] + 1
	}

	sl2 <- csl/gen
	return(list(s=sl, mean=sl2, dist=occur, log=log))
}

kqr_FO_sim = function(files, reader=files2dur) {
	qs <- reader(files)
	return(kqr_FO(qs))
}

kqr_FO = function(qs, scale=100) {
	win <- scale * sum(qs)
	fsched <- function(occur, w=win, coef=qs) {
		ret <- sum(w/(coef * occur)) + w*(sum(coef * occur)-3*w)^2; 
		return(ret)
	}

	initv <- rep(1, length(qs))
	res <- spg(initv, fsched, lower=initv, upper=win/qs, control=list(maximize=FALSE, trace=FALSE))

	occur <- res$par
	sl <- sum((win/qs)/occur) /length(qs)

	return(list(s=max(1,sl), dist=occur, span=win))
}


function greedy(I::Instance, iterations::Int64=100)
  #=
  Greedy heuristic
  =#
  tic = time()
  U = zeros(Int64,I.nb_prod)
  marginal = 0
  for i = 1:I.capacity
    prod = 0
    #println(U)
    for j = 1:I.nb_prod
      if (U[j] < I.M)
        U2 = copy(U)
        U2[j] = U2[j] + 1
        # local_marginal = approximate_revenue(I,U2,iterations)
        local_marginal = eval_revenue(I,U2,iterations)
        if local_marginal > marginal
          marginal = local_marginal
          prod = j
        end
      end
    end
    if prod > 0
      U[prod] = U[prod] + 1
    end
  end
  # return(approximate_revenue(I,U,500),time() - tic)
  return(eval_revenue(I,U),time() - tic)
end


function local_swaps(I::Instance, iterations::Int64=100)
  #=
  Local search heuristic
  =#
  tic = time()
  U = zeros(Int64,I.nb_prod)
  U[div(I.nb_prod,2)] = I.capacity
  marginal = 0
  gain = 1
  step = 0
  while (gain > 0.005) & (step < I.capacity) & (time() - tic < 600.)
    prod = (0,0)
    step += 1
    marginal_initiale = marginal
    #println(U)
    for i in find(U)
      for j in 1:I.nb_prod
        if (U[j] < I.M) & (i != j)
          U2 = copy(U)
          U2[j] = U2[j] + 1
          U2[i] = U2[i] - 1
          # local_marginal = approximate_revenue(I,U2,iterations)
          local_marginal = eval_revenue(I,U2,iterations)
          if local_marginal > marginal
            marginal = local_marginal
            prod = (i,j)
          end
        end
      end
    end
    if prod[1] > 0
      U[prod[2]] = U[prod[2]] + 1
      U[prod[1]] = U[prod[1]] - 1
      gain = marginal/marginal_initiale-1.
    end
  end
  # return(approximate_revenue(I,U,500),time() - tic)
  return(eval_revenue(I,U),time() - tic)
end


function naming(x::Array{Float64,1})
  return(join(map(y -> string(y),x),"-"))
end


function lookup_sample(I,sol::Array{Float64,1},dict::Dict{Any,Any},iterations::Int64)
    name = naming(sol)
    if ~haskey(dict,name)
        # dict[name] = approximate_revenue(I,map(x -> Int(x),sol),iterations)
        dict[name] = eval_revenue(I,map(x -> Int(x),sol),iterations)
      end
    return(name)
end


function lovaszc_greedy(I::Instance, iterations::Int64=100)
  #=
  Continuous greedy heuristic
  =#
  tic = time()
  solution_local = zeros(Float64,I.nb_prod)
  #solution_local[div(I.nb_prod,2)] = I.capacity-1

  val_dict = Dict()

  #println(naming(solution_local))

  val_dict[naming(solution_local)] =  0.0

  eps = 0.05*I.capacity

  e = [i => zeros(Float64,I.nb_prod) for i = 1:I.nb_prod]

  for i = 1:I.nb_prod
      e[i][i] = 1.0
  end

  marginal_increase = 1.0
  iteration = 0

  while ((sum(solution_local) < float(I.capacity)) | (marginal_increase > 0.005)) & (iteration < 2000) & (time() - tic < 600.)
      iteration = iteration + 1
      solution_base = floor(solution_local)
      ordered_set = sortperm(solution_local-solution_base)
      #ordered_val = [solution_local[ordered_set[0]]-solution_base[ordered_set[0]]]
      #ordered_val = ordered_val + map(lambda x: solution_local[ordered_set[x]]-solution_local[ordered_set[x-1]],range(1,self.n))
      vectors = zeros(Float64,I.nb_prod+1,I.nb_prod)
      for u = 1:I.nb_prod
        vectors[u,:] = reduce((x,y) -> x + e[y], copy(solution_base),ordered_set[u:I.nb_prod])
      end
      #vectors = map(u -> ,1:I.nb_prod)
      vectors[I.nb_prod+1,:] = solution_base
      #println(vectors)
      names = mapslices(x -> lookup_sample(I,x,val_dict,iterations),vectors,2)
      #println(val_dict)
      gradients = map(x -> val_dict[names[x]] - val_dict[names[x+1]],1:I.nb_prod)
      gradient = map(x ->gradients[find(ordered_set.==x)[1]],1:I.nb_prod)
      eps = max((float(I.capacity) - sum(solution_local))/2.,0.05*float(I.capacity))
      solution_local = max(solution_local + eps*1./sum(abs(gradient))*gradient,0.)
      #println(names)
      marginal_increase = eps/sum(abs(gradient))/(0.001+val_dict[names[I.nb_prod+1]])*sum(gradient.*gradient)
      if sum(solution_local) > float(I.capacity)
        solution_local = float(I.capacity)/sum(solution_local)*solution_local
        marginal_increase = float(I.capacity)/sum(solution_local)*marginal_increase
      end
      name_new = lookup_sample(I,floor(solution_local),val_dict,iterations)
      #println(marginal_increase)
      #println(floor(solution_local))
    end

  solution_local = I.capacity/sum(solution_local)*solution_local

  solution = floor(solution_local)

  #Use spare capacity due to rounding to improve the expected revenue
  if sum(solution) < I.capacity
      indices = sortperm(solution_local - solution,rev= true)
      for i = 1:(I.capacity - Int(sum(solution)))
          ind = indices[i]
          solution[ind] = solution[ind] + 1
      end
  end
  # return(approximate_revenue(I,map(x -> Int(x),solution),500),time() - tic)
  try
    return(eval_revenue(I,map(x -> Int(floor(x)),solution)),time() - tic)
  catch e
    bt = catch_backtrace()
    msg = sprint(showerror, e, bt)
    println(msg)
    solution = ones(Float64,I.nb_prod)
    solution = I.capacity/sum(solution)*solution
    solution = floor(solution)
    return(eval_revenue(I,map(x -> Int(floor(x)),solution)),time() - tic)
  end
end


function intervals_approx(I::Instance,iterations::Int64=100)
  #=
  Approximation algorithm for the intervals choice model
  =#
  tic = time()

  probas = zeros(Float64,(I.nb_prod+1,I.nb_prod))

  # pre-computation of probas
  for i1 in 1:I.nb_prod
    for i2 in 0:(i1-1)
      if i2>0
        already_captured = max(0,1.0.*I.consideration_sets[:,i2])
      else
        already_captured = zeros(Float64,I.nb_cst_type)
      end
      new_capture = max(0,1.0.*I.consideration_sets[:,i1] - already_captured)
      probas[1+i2,i1] = vecdot(new_capture,I.lambdas)
    end
  end
  # pre-computation of binomial coefs
  B = zeros(I.M,I.nb_prod+1,I.nb_prod,I.M)
  for alpha = 1:I.M
    for beta1 = 1:I.nb_prod
      for beta2 in 0:(beta1-1)
        for gamma = 1:alpha
          B[alpha,1+beta2,beta1,gamma] = pdf(Binomial(alpha,probas[1+beta2,beta1]),gamma)
        end
      end
    end
  end

  # pre-computation of demand
  t = zeros(I.M)
  t[2:I.M] = I.demand_cdf[1:(I.M-1)]
  demand_pdf = I.demand_cdf - t

  # DP computation
  DP_val = zeros(Float64,(I.nb_prod+1,I.nb_prod,I.capacity+1))
  assortment_action = zeros(Int64,(I.nb_prod+1,I.nb_prod,I.capacity+1))
  capacity_action = zeros(Float64,(I.nb_prod+1,I.nb_prod,I.capacity+1))
  for i1 in 1:I.nb_prod
    for i2 in 0:(i1-1)
      for c in 1:I.capacity
        if (i1>1) & (i2>0)
          r = zeros(Float64,(1+c,1+i2-1))
          for i3 in 0:(i2-1)
            for c_local in 0:c
                r[1+c_local,1+i3] = (sum([(demand_pdf[k])*vecdot(min(c_local,1:k),I.prices[i1]*
                                    B[k,1+i2,i1,1:k]) for k = 1:I.M]) +
                                      DP_val[1+i3,i2,1+c-c_local])
            end

          end
        else
          r = zeros(Float64,1+c)
          for c_local in 0:c
              r[1+c_local] = (sum([(demand_pdf[k])*vecdot(min(c_local,1:k),I.prices[i1]*
                                  B[k,1+i2,i1,1:k]) for k = 1:I.M]))
          end
        end
        best_choice = findmax(r)
        if (i1>1) & (i2>0)
          assortment_action[1+i2,i1,1+c] = Int(ceil(best_choice[2]/(1.0+c)))-1
          capacity_action[1+i2,i1,1+c] = ((best_choice[2]-1)% (1+c))
        else
          # assortment_action[1+i2,i1,1+c] = 0
          capacity_action[1+i2,i1,1+c] = (best_choice[2]-1)
        end
        DP_val[1+i2,i1,1+c] = best_choice[1]
      end
    end
  end


  # chooses inventory allocation
  U = zeros(Int64,I.nb_prod)
  c = I.capacity
  i1 = I.nb_prod
  i2 = findmax(DP_val[:,i1,1+c])[2]-1
  while (i1 > 0) & (c> 0)
    U[i1] = capacity_action[1+i2,i1,1+c]
    i1_prev = i1
    i1 = i2
    i2 = assortment_action[1+i2,i1_prev,1+c]
    c = c - U[i1_prev]
  end
  # println(U)
  # rev = 0
  # i2 = 0
  # for i in find(U.>0)
  #   println(i,"-",sum([(demand_pdf[k])*vecdot(min(U[i],1:k),I.prices[i]*
  #                       B[k,1+i2,i,1:k]) for k = 1:I.M]))
  #   rev = rev + sum([(demand_pdf[k])*vecdot(min(U[i],1:k),I.prices[i]*
  #                       B[k,1+i2,i,1:k]) for k = 1:I.M])
  #   i2 = i
  # end
  # println(rev)
  return(eval_revenue(I,U),time()-tic)
end

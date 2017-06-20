function greedy(I::Instance, iterations::Int64=100)
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
        #local_marginal = approximate_revenue(I,U2,iterations)
        local_marginal = exact_revenue(I,U2)
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
  #return(approximate_revenue(I,U,500),time() - tic)
  return(exact_revenue(I,U),time() - tic)
end

function local_swaps(I::Instance, iterations::Int64=100)
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
          #local_marginal = approximate_revenue(I,U2,iterations)
          local_marginal = exact_revenue(I,U2)
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
  #return(approximate_revenue(I,U,500),time() - tic)
  return(exact_revenue(I,U),time() - tic)
end

function naming(x::Array{Float64,1})
  return(join(map(y -> string(y),x),"-"))
end

function lookup_sample(I,sol::Array{Float64,1},dict::Dict{Any,Any},iterations::Int64)
    name = naming(sol)
    if ~haskey(dict,name)
        #dict[name] = approximate_revenue(I,map(x -> Int(x),sol),iterations)
        dict[name] = exact_revenue(I,map(x -> Int(x),sol))
      end
    return(name)
end

function lovaszc_greedy(I::Instance, iterations::Int64=100)
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
      #println(iteration)
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
  #return(approximate_revenue(I,map(x -> Int(x),solution),500.),time() - tic)
  return(exact_revenue(I,map(x -> Int(x),solution)),time() - tic)
end


function horizontal_approx(I::Instance,threshold::Int64)
  already_captured = zeros(Float64,(I.nb_cst_type,1))
  remaining_set = Array(threshold:I.nb_prod)

  # Solving the static problem greedily
  for i = 1:I.capacity
    incr_save = 0
    new_capture_save = zeros(Float64,(I.nb_cst_type,1))
    prod_save = 0
    for j in remaining_set
      new_capture = max(0,1.0.*I.consideration_sets[:,j] -already_captured)
      incr = vecdot(new_capture,I.lambdas)
      if incr >= incr_save
        incr_save = incr
        new_capture_save = new_capture
        prod_save = j
      end
    end
    if incr_save > 0.
      already_captured = already_captured + new_capture_save
      remaining_set = setdiff(remaining_set,prod_save)
    else
      break
    end
  end

  # Optimal assortment
  assortment = setdiff(Array(threshold:I.nb_prod),remaining_set)
  assortment_bin = zeros(1,I.nb_prod)
  assortment_bin[assortment] = 1.

  # List to product assignment
  A = (ones((I.nb_cst_type,1))*assortment_bin).*I.rankings
  assignment = transpose((findmax(transpose(A),1)[2]-1)%(I.nb_prod)+1)
  probas = max(0.0,min(map( x -> sum(I.lambdas[find(assignment.== x)]),assortment),1.0))
  # Solving the newsvendor greedily
  U = zeros(Int64,I.nb_prod)
  t = zeros(I.M)
  t[2:I.M] = I.demand_cdf[1:(I.M-1)]
  demand_pdf = I.demand_cdf - t
  #println(demand_pdf)
  # preparing the binomial coefficients
  B = zeros(I.M,size(assortment,1),I.M)
  for alpha = 1:I.M
    for beta = 1:size(assortment,1)
      for gamma = 1:alpha
        B[alpha,beta,gamma] = pdf(Binomial(alpha,probas[beta]),gamma)
      end
    end
  end

  for i = 1:I.capacity
    marginal = 0
    prod = 0
    for (a,j) in enumerate(assortment)
      if (U[j] < I.M)
        local_marginal = sum([(demand_pdf[k])*sum(B[k,a,(U[j]+1):k]) for k = (U[j]+1):I.M])
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
  #alpha = approximate_revenue(I,U,500)
  alpha = exact_revenue(I,U)
  return(alpha)
end

function general_approx(I::Instance)
  tic = time()
  return(maximum(x-> horizontal_approx(I,x),1:I.nb_prod),time() - tic)
end

function nested_approx(I::Instance, iterations::Int64=100)
  tic = time()
  sel =[ (threshold - 1 +findmax([sum(I.lambdas[j:I.nb_prod]) for j=threshold:I.nb_prod].*I.prices[threshold:I.nb_prod])[2]) for threshold = 1:I.nb_prod ]
  #println(sel)
  sel = unique(sel)
  U = zeros(Int64,I.nb_prod)
  marginal = 0

  for i = 1:I.capacity
    prod = 0
    #println(U)
    for j in sel
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
    else
      #Add one unit to most expensive if (because of the approximate estimes) there is no marginal increase
      U[I.nb_prod] = U[I.nb_prod] + 1
    end
  end
  #return(approximate_revenue(I,U,500),time() - tic)
  return(exact_revenue(I,U),time() - tic)
end

function goyal_approx(I::Instance, iterations::Int64=100)
  tic = time()
  #Fix eps
  eps = 0.3

  #Identify frequent + rare
  psi = sum((I.lambdas*ones(1,I.nb_prod)).*I.consideration_sets,1)
  i = I.nb_prod + 1
  test = 0
  while (test == 0) & (i > 0)
    i = i - 1
    tot = mean(Binomial(1,psi[i]))
    for m  = 2:I.M
      tot = tot + (1-I.demand_cdf[m-1])*(mean(Binomial(m,psi[i])) - mean(Binomial(m-1,psi[i])))
    end
    if tot > eps*eps*I.capacity
      test = 1
    end
  end

  #Pick the best static rare item
  i = i + 1
  if i <= I.nb_prod
    i_star = findmax(I.prices[i:I.nb_prod].*psi[i:I.nb_prod])[2]
  else
    i_star = i-1
  end
  i_collection = [i_star]
  #println(i_star,i)
  #Bucket frequent items
  if i > 1
    i_F = i - 1
    thresh = eps*eps*eps*I.prices[i_F]
    while thresh < I.prices[i_F]
      next = findmax(I.prices .> thresh)[2]
      next2 = findmax(I.prices .> (1+eps)*thresh)[2]
      if ((next2 == 1) & (next != 1)) | (next2 >= i_F)
        i_collection = union(i_collection,[i_F])
      else
        i_collection = union(i_collection,[max(next2-1,1)])
      end
      #i_collection = union(i_collection,[next])
      thresh = thresh*(1.+eps)
    end
  end

  #Enumerate over vectors
  #i_collection = sort(i_collection)[(size(i_collection,1)-4):size(i_collection,1)]
  i_collection = sort(i_collection, rev = true)
  val_max = 0.
  U_max = zeros(Int64,I.nb_prod)
  max_exponent = Int64(floor(log(I.capacity)/log(1+eps)))
  iterator_spe = reduce((x,y)-> Base.product(0:max_exponent,x),0:max_exponent,1:(size(i_collection,1)-1))
  for v in iterator_spe
    if time() - tic > 600
      break
    end
    U = vec_create(I,rec(v),i_collection,eps)
    #println(U)
    if sum(U) <= I.capacity
      V = floor(U/(sum(U)+0.000001)*I.capacity)
      V[findmax(U)[2]] = V[findmax(U)[2]] + I.capacity - sum(V)
      U = convert(Array{Int64},V)
      #rev = approximate_revenue(I,U,iterations)
      rev = exact_revenue(I,U)
      #println(U,rev)
      if  rev > val_max
        U_max = U
        val_max = rev
      end
    end
  end
  #return(approximate_revenue(I,U_max,500),time() - tic)
  return(exact_revenue(I,U_max),time() - tic)
end

function vec_create(I::Instance, A::Array{Int64},i_collection::Array{Int64,1},eps::Float64)
  d = size(A,1)
  U_vec = zeros(Int64,I.nb_prod)
  for i=1:d
    if A[i] == 0
      U_vec[i_collection[i]] = 0
    else
      U_vec[i_collection[i]] = Int64(floor((1+eps)^(A[i])))
    end
  end
  return(U_vec)
end

function rec(x)
  if typeof(x)== Int64
    return([x[1]])
  else
    return(vcat([x[1]],rec(x[2])))
  end
end

##############################################################################
# Generates the instance
##############################################################################

type Instance
  nb_prod::Int64
  nb_cst_type::Int64
  M::Int64
  capacity::Int64
  demand_cdf::Vector{Float64}
  prices::Vector{Float64}
  lambdas::Vector{Float64}
  consideration_sets::BitArray{2}
  rankings::Array{Float64,2}
  solution::Vector{Int64}
  Instance(np,nc,nM,c) = new(Int64(np),Int64(nc),Int64(nM),Int64(c),
                           zeros(Float64,nM),
                           zeros(Float64,np),zeros(Float64,nc),
                           zeros(Bool,nc,np),zeros(Float64,nc,np),
                           zeros(Int64,np));
end

function Initialization_cdf!(I::Instance, distribution_type::String)
  if distribution_type == "Normal"
    I.demand_cdf = cumsum(pdf(Truncated(Normal(30,40),1,I.M),1:I.M))
    I.demand_cdf = I.demand_cdf/I.demand_cdf[I.M]
  elseif distribution_type == "Geometric"
    I.demand_cdf = cumsum(pdf(Truncated(Geometric(0.02),1,I.M),1:I.M))
    I.demand_cdf = I.demand_cdf/I.demand_cdf[I.M]
  elseif distribution_type == "NonIFRdiscrete"
    support = sort(sample(Array(1:I.M),5,replace = false))
    I.demand_cdf[support] = Array(1:5)/5.0
    for i = 2:I.M
      I.demand_cdf[i] = maximum(I.demand_cdf[1:i])
    end
    #println(I.demand_cdf)
  elseif distribution_type == "NonIFR"
    failure_rates = 6.0/float(I.M)*rand(I.M)
    failure_rates[I.M] = 1.
    for j = 1:I.M
      if j == 1
        I.demand_cdf[j] = failure_rates[j]
      else
        I.demand_cdf[j] = failure_rates[j]*(1-I.demand_cdf[j-1]) + I.demand_cdf[j-1]
      end
    end
  else
    failure_rates = sort(6.0/float(I.M)*rand(I.M))
    failure_rates[I.M] = 1.
    for j = 1:I.M
      if j == 1
        I.demand_cdf[j] = failure_rates[j]
      else
        I.demand_cdf[j] = failure_rates[j]*(1-I.demand_cdf[j-1]) + I.demand_cdf[j-1]
      end
    end
  end
end

function Initialization!(I::Instance, model_type::String, bernouilli_param::Float64 = 0.35)
  #Initializes the intance, filling with numerical values
  randn!(I.prices)
  I.prices = sort(exp(I.prices))
  randn!(I.lambdas)
  I.lambdas = exp(I.lambdas)
  I.lambdas = I.lambdas/sum(I.lambdas)

  if model_type == "General"
    I.consideration_sets = rand(I.nb_cst_type,I.nb_prod) .< bernouilli_param
    for i = 1:I.nb_cst_type
      I.rankings[i,:] = map(x -> min(Float64(x[2]),Float64(I.nb_prod*I.consideration_sets[i,x[1]])),
                            enumerate(shuffle(Array(1:I.nb_prod)))
                            )
    end

  elseif model_type == "Nested"
    for j = 1:min(I.nb_cst_type,I.nb_prod)
      for i = 1:I.nb_prod
        if i <= j
          I.consideration_sets[j,i] = 1
        end
      end
    end

    for i = 1:I.nb_cst_type
      I.rankings[i,:] = I.nb_prod - Array(1:I.nb_prod) + 1
    end

    if I.nb_prod < I.nb_cst_type
      I.lambdas[(I.nb_prod+1):I.nb_cst_type] = 0
      I.lambdas[1:I.nb_prod] = I.lambdas[1:I.nb_prod]/sum(I.lambdas[1:I.nb_prod])
    end

    ##Non IFR distribution
    # failure_rates = 3./float(I.M)*rand(I.M)
    # failure_rates[I.M] = 1.
    # for j = 1:I.M
    #   if j == 1
    #     I.demand_cdf[j] = failure_rates[j]
    #   else
    #     I.demand_cdf[j] = failure_rates[j]*(1-I.demand_cdf[j-1]) + I.demand_cdf[j-1]
    #   end
    # end

  elseif model_type == "Intervals"
    for j = 1:I.nb_cst_type
      ia = sample(1:I.nb_prod)
      ib = sample(ia:I.nb_prod)
      for i = ia:ib
          I.consideration_sets[j,i] = 1
      end
    end
    for i = 1:I.nb_cst_type
      I.rankings[i,:] = I.nb_prod - Array(1:I.nb_prod) + 1
    end
  end

end

function exact_revenue(I::Instance, U::Vector{Int64})
   current_capacity = sum(U)
   prod_of_unit = zeros(Int64,current_capacity)
   for i = 1:current_capacity
     prod_of_unit[i] = findmax(cumsum(U).>= i)[2]
   end
   probas = zeros(I.M,current_capacity)
   probas[1,1] = 1.0
   psi_vec = zeros(current_capacity-1)
   for i = 1:(current_capacity-1)
     psi_vec[i] = sum(I.lambdas[prod_of_unit[i]:I.nb_cst_type])
   end
   mat = diagm(psi_vec,1)
   for i = 1:(current_capacity-1)
     mat[i,i] = 1-mat[i,i+1]
   end
   alpha = sum(I.lambdas[prod_of_unit[current_capacity]:I.nb_cst_type])
   mat[current_capacity,current_capacity] = 1 - alpha
   rev = I.prices[prod_of_unit[1]]*vcat(psi_vec,[alpha])[1]
   #println(size(mat))
   #println(mat)
   #println(psi_vec)
   for i = 2:I.M
     probas[i,:] = transpose(probas[i-1,:])*mat
     #println(probas[i,:])
     rev = rev + (1-I.demand_cdf[i-1])*dot(probas[i,:],vcat(psi_vec,[alpha]).*I.prices[prod_of_unit])
   end
   return(rev)
end

function sample_revenue(I::Instance,U::Vector{Int64})
  for i = 1:I.nb_prod
    I.solution[i] = U[i]
  end
  M = findmax(I.demand_cdf .> rand())[2]
  model = Multinomial(1,I.lambdas)
  arrivals = rand(model,M)
  assortment = reshape(I.solution .> 0,(I.nb_prod,))
  revenue = 0
  for j = 1:M
    ctype = findfirst(arrivals[:,j])
    prod = findmax(I.rankings[ctype,:] .* I.consideration_sets[ctype,:].* assortment)[2]
    # if prod == 21
    #   println(size(assortment))
    #   println(size(I.rankings[ctype,:]))
    #   println(I.rankings[ctype,:] .* assortment)
    #   println(findmax(I.rankings[ctype,:] .* assortment))
    # end
    if (I.consideration_sets[ctype,prod] > 0) &(I.rankings[ctype,prod] > 0) & (assortment[prod])
      I.solution[prod] = I.solution[prod]-1
      revenue = revenue + I.prices[prod]
    end
    if I.solution[prod] == 0
      assortment[prod] = false
    end
  end
  return(revenue)
end

function approximate_revenue(I::Instance,U::Vector{Int64},iterations::Int64=100)
  return(sum(map(x -> sample_revenue(I,U), 1:iterations))/iterations)
end

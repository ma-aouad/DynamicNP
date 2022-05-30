using NPZ,Distributions,Base.Cartesian

include("instancegen.jl")
include("heuristics.jl")
include("intervals.jl")


function run_experiments(distribution_type::String,model_type::String,logs::Bool=false)
  #=
  Example of code to run the methods
  =#
  M = 100
  prods = [20]
  types = [160]
  capacities = [20,40,80,150]
  num_instances = 10

  time = zeros(size(prods)[1],size(types)[1],size(capacities)[1],6,num_instances)
  sol = zeros(size(prods)[1],size(types)[1],size(capacities)[1],6,num_instances)

  for (i_n,n) in enumerate(prods)
    for (i_K,K) in enumerate(types)
      for (i_C,C) in enumerate(capacities)
        for iteration = 1:num_instances

          println(n,"-",K,"-",C,"-",iteration)

          I = Instance(n,K,M,C)
          Initialization_cdf!(I,distribution_type)
          Initialization!(I,model_type)
          if logs
            log_instance(I,distribution_type,model_type,string(K,'-',C,'-',iteration))
          end

          output = general_approx(I)
          println("GA",output)
          time[i_n,i_K,i_C,1,iteration] = output[2]
          sol[i_n,i_K,i_C,1,iteration] = output[1]

          output = greedy(I)
          time[i_n,i_K,i_C,2,iteration] = output[2]
          sol[i_n,i_K,i_C,2,iteration] = output[1]
          println("DG",output)

          output = lovaszc_greedy(I)
          time[i_n,i_K,i_C,3,iteration] = output[2]
          sol[i_n,i_K,i_C,3,iteration] = output[1]
          println("GD",output)

          output = local_swaps(I)
          time[i_n,i_K,i_C,4,iteration] = output[2]
          sol[i_n,i_K,i_C,4,iteration] = output[1]
          println("LS",output)

          if model_type == "Nested"
            output = nested_approx(I)
            time[i_n,i_K,i_C,5,iteration] = output[2]
            sol[i_n,i_K,i_C,5,iteration] = output[1]
            println("NE",output)

            output = goyal_approx(I)
            time[i_n,i_K,i_C,6,iteration] = output[2]
            sol[i_n,i_K,i_C,6,iteration] = output[1]
            println("GO",output)

          end
          if logs
            npzwrite(string("outputs/I-time",distribution_type,model_type,".npz"),time)
            npzwrite(string("outputs/I-sol",distribution_type,model_type,".npz"),sol)
          end
        end

      end
    end
  end
end

run_experiments("Normal","General")
run_experiments("Poisson","General")
run_experiments("Geometric","General")

run_experiments("Normal","Intervals")
run_experiments("Poisson","Intervals")
run_experiments("Geometric","Intervals")

run_experiments("Normal","Nested")
run_experiments("Poisson","Nested")
run_experiments("Geometric","Nested")

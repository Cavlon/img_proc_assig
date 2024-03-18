import pstats

p = pstats.Stats('profile_result.prof')
p.sort_stats('cumulative').print_stats(10)  # Print the top 10 functions

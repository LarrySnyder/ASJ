import cProfile
import pstats
import io

import predpol_sim



pr = cProfile.Profile()
pr.enable()

cProfile.run('predpol_sim.main()', 'profile_results')

pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats()

with open('predpol/profile_results.txt', 'w+') as f:
    f.write(s.getvalue())
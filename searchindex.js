Search.setIndex({"docnames": ["LOSC_Event_tutorial", "WORKING_LOSC_Event_tutorial", "contribution_statement", "hw04_description"], "filenames": ["LOSC_Event_tutorial.ipynb", "WORKING_LOSC_Event_tutorial.ipynb", "contribution_statement.md", "hw04_description.md"], "titles": ["BINARY BLACK HOLE SIGNALS IN LIGO OPEN DATA", "BINARY BLACK HOLE SIGNALS IN LIGO OPEN DATA", "&lt;no title&gt;", "Homework No 4 - <em>From Notebooks to Research Packages, Part II</em>"], "terms": {"repositori": [0, 1], "public": [0, 1, 3], "so": [0, 1, 3], "binder": [0, 1, 3], "can": [0, 1, 3], "all": [0, 1, 3], "base": [0, 1], "origin": [0, 1], "center": [0, 1], "scienc": [0, 1], "class": [0, 1, 3], "exercis": [0, 1], "restructur": [0, 1], "improv": [0, 1, 3], "reproduc": [0, 1, 3], "homework": [0, 1], "assign": [0, 1, 3], "spring": [0, 1, 3], "2023": [0, 1, 3], "uc": [0, 1, 3], "berkelei": [0, 1, 3], "s": [0, 1], "stat": [0, 1], "159": [0, 1, 3], "259": [0, 1, 3], "cours": [0, 1], "authorship": [0, 1, 3], "rest": [0, 1], "version": [0, 1], "1": [0, 1], "64": [0, 1], "2022": [0, 1, 3], "juli": [0, 1], "05": [0, 1], "welcom": [0, 1], "ipython": [0, 1], "notebook": [0, 1], "associ": [0, 1], "script": [0, 1], "losc_event_tutori": [0, 1], "py": [0, 1], "go": [0, 1, 3], "through": [0, 1], "some": [0, 1], "typic": [0, 1], "task": [0, 1], "strain": [0, 1], "time": [0, 1], "seri": [0, 1], "releas": [0, 1], "losc": [0, 1], "eventapi": [0, 1], "run": [0, 1, 3], "link": [0, 1, 3], "page": [0, 1], "If": [0, 1], "you": [0, 1, 3], "your": [0, 1, 3], "own": [0, 1], "section": [0, 1], "below": [0, 1], "after": [0, 1], "desir": [0, 1], "eventnam": [0, 1], "just": [0, 1], "full": [0, 1], "question": [0, 1], "suggest": [0, 1], "correct": [0, 1], "etc": [0, 1, 3], "ask": [0, 1], "igwn": [0, 1], "assum": [0, 1], "comfort": [0, 1], "also": [0, 1], "know": [0, 1], "bit": [0, 1, 3], "about": [0, 1, 3], "digit": [0, 1], "want": [0, 1, 3], "learn": [0, 1], "includ": [0, 1, 3], "power": [0, 1], "manipul": [0, 1], "vast": [0, 1], "complex": [0, 1], "topic": [0, 1], "we": [0, 1, 3], "cover": [0, 1], "mani": [0, 1], "basic": [0, 1, 3], "beginn": [0, 1], "resourc": [0, 1], "web": [0, 1], "101scienc": [0, 1], "com": [0, 1], "dsp": [0, 1], "htm": [0, 1], "coursera": [0, 1], "georgemdalla": [0, 1], "wordpress": [0, 1], "2014": [0, 1], "14": [0, 1, 3], "wavelet": [0, 1], "4": [0, 1], "dummi": [0, 1], "fourier": [0, 1], "transform": [0, 1], "heisenberg": [0, 1], "en": [0, 1, 3], "wikipedia": [0, 1, 3], "wiki": [0, 1, 3], "signal_process": [0, 1], "spectral_dens": [0, 1], "greenteapress": [0, 1], "thinkdsp": [0, 1], "digital_filt": [0, 1], "And": [0, 1], "well": [0, 1], "lot": [0, 1], "more": [0, 1, 3], "pre": [0, 1], "configur": [0, 1, 3], "setup": [0, 1, 3], "eg": [0, 1], "great": [0, 1], "don": [0, 1], "t": [0, 1], "have": [0, 1, 3], "up": [0, 1, 3], "anyth": [0, 1], "otherwis": [0, 1], "begin": [0, 1, 3], "get": [0, 1, 3], "necessari": [0, 1], "zip": [0, 1], "unpack": [0, 1], "singl": [0, 1], "directori": [0, 1, 3], "ipynb": [0, 1], "readligo": [0, 1], "32": [0, 1], "4096": [0, 1], "hz": [0, 1], "hdf5": [0, 1], "format": [0, 1], "both": [0, 1], "detector": [0, 1, 3], "plu": [0, 1], "cross": [0, 1], "polar": [0, 1], "paramet": [0, 1], "need": [0, 1, 3], "few": [0, 1], "numpi": [0, 1], "matplotlib": [0, 1], "scipi": [0, 1], "h5py": [0, 1], "hint": [0, 1], "tutorial00": [0, 1], "should": [0, 1, 3], "work": [0, 1, 3], "2": [0, 1], "6": [0, 1], "abov": [0, 1, 3], "3": [0, 1], "recent": [0, 1], "might": [0, 1], "warn": [0, 1], "futurewarn": [0, 1], "messag": [0, 1, 3], "tend": [0, 1], "differ": [0, 1], "hopefulli": [0, 1], "thei": [0, 1], "ignor": [0, 1, 3], "filetyp": [0, 1], "mean": [0, 1], "hdfgroup": [0, 1], "gp": [0, 1], "number": [0, 1, 3], "second": [0, 1], "sinc": [0, 1, 3], "jan": [0, 1], "1980": [0, 1], "gmt": [0, 1], "oc": [0, 1], "np": [0, 1], "edu": [0, 1], "oc2902w": [0, 1], "timsi": [0, 1], "html": [0, 1, 3], "me": [0, 1], "most": [0, 1], "default": [0, 1], "select": [0, 1], "must": [0, 1], "one": [0, 1, 3], "proce": [0, 1], "gw150914": [0, 1], "gw151226": [0, 1], "lvt151012": [0, 1], "gw170104": [0, 1], "make_plot": [0, 1], "plottyp": [0, 1], "png": [0, 1], "pdf": [0, 1], "standard": [0, 1], "numer": [0, 1], "import": [0, 1, 3], "interpol": [0, 1], "interp1d": [0, 1], "butter": [0, 1], "filtfilt": [0, 1], "iirdesign": [0, 1], "zpk2tf": [0, 1], "freqz": [0, 1], "magic": [0, 1], "out": [0, 1], "doesn": [0, 1], "inlin": [0, 1], "config": [0, 1], "inlinebackend": [0, 1], "figure_format": [0, 1], "retina": [0, 1], "pyplot": [0, 1], "plt": [0, 1], "mlab": [0, 1], "specif": [0, 1], "ligotool": [0, 1, 3], "rl": [0, 1], "fnjson": [0, 1], "bbh_events_v3": [0, 1], "try": [0, 1], "load": [0, 1], "r": [0, 1], "except": [0, 1], "ioerror": [0, 1], "print": [0, 1], "cannot": [0, 1], "quit": [0, 1], "did": [0, 1], "user": [0, 1], "an": [0, 1, 3], "extract": [0, 1], "fn_h1": [0, 1], "fn_l1": [0, 1], "fn_templat": [0, 1], "fs": [0, 1], "tevent": [0, 1], "approxim": [0, 1], "fband": [0, 1], "band": [0, 1], "bandpass": [0, 1], "nameerror": 0, "traceback": 0, "call": [0, 3], "last": 0, "input": 0, "In": [0, 1, 3], "cell": [0, 1, 3], "line": [0, 1], "defin": [0, 1], "filenam": [0, 1], "strain_h1": [0, 1], "time_h1": [0, 1], "chan_dict_h1": [0, 1], "loaddata": [0, 1], "strain_l1": [0, 1], "time_l1": [0, 1], "chan_dict_l1": [0, 1], "them": [0, 1], "gener": [0, 1, 3], "ha": [0, 1, 3], "fill": [0, 1], "nan": [0, 1], "when": [0, 1], "take": [0, 1], "valid": [0, 1], "qualiti": [0, 1], "analyz": [0, 1], "requir": [0, 1], "loop": [0, 1], "over": [0, 1], "stretch": [0, 1], "simplic": [0, 1], "detail": [0, 1], "same": [0, 1, 3], "vector": [0, 1], "interv": [0, 1], "uniformli": [0, 1], "dt": [0, 1], "0": [0, 1], "let": [0, 1, 3], "stuff": [0, 1], "len": [0, 1], "min": [0, 1], "max": [0, 1], "what": [0, 1], "chan_dict": [0, 1], "usabl": [0, 1, 3], "sum": [0, 1], "131072": [0, 1], "1126259446": [0, 1], "1126259461": [0, 1], "999878": [0, 1], "1126259477": [0, 1], "9997559": [0, 1], "7": [0, 1], "044665943156067e": [0, 1], "19": [0, 1], "5": [0, 1], "895522509246437e": [0, 1], "23": [0, 1], "706262192397465e": [0, 1], "8697138664279764e": [0, 1], "18": [0, 1], "0522332249909908e": [0, 1], "60035111311666e": [0, 1], "20": [0, 1], "deltat": [0, 1], "around": [0, 1], "index": [0, 1], "indxt": [0, 1], "where": [0, 1], "figur": [0, 1, 3], "label": [0, 1], "g": [0, 1], "xlabel": [0, 1], "str": [0, 1], "ylabel": [0, 1], "legend": [0, 1], "loc": [0, 1], "lower": [0, 1], "right": [0, 1], "titl": [0, 1], "savefig": [0, 1], "_strain": [0, 1], "1126259462": [0, 1], "44": [0, 1], "domin": [0, 1], "low": [0, 1], "nois": [0, 1], "wai": [0, 1], "without": [0, 1, 3], "domain": [0, 1], "give": [0, 1, 3], "idea": [0, 1], "A": [0, 1], "visual": [0, 1], "squar": [0, 1], "root": [0, 1], "psd": [0, 1, 3], "averag": [0, 1], "fast": [0, 1], "fft": [0, 1], "estim": [0, 1], "equival": [0, 1], "versu": [0, 1], "limit": [0, 1], "abil": [0, 1], "identifi": [0, 1], "gw": [0, 1], "unit": [0, 1], "rt": [0, 1], "rm": [0, 1], "integr": [0, 1, 3], "There": [0, 1, 3], "moment": [0, 1], "make_psd": [0, 1], "nfft": [0, 1], "pxx_h1": [0, 1], "freq": [0, 1], "pxx_l1": [0, 1], "psd_h1": [0, 1], "psd_l1": [0, 1], "smooth": [0, 1], "dure": [0, 1], "o1": [0, 1], "ll": [0, 1], "later": [0, 1], "pxx": [0, 1], "e": [0, 1], "22": [0, 1], "7e": [0, 1], "2000": [0, 1], "psd_smooth": [0, 1], "overlaid": [0, 1], "f_min": [0, 1], "f_max": [0, 1], "figsiz": [0, 1], "10": [0, 1], "8": [0, 1], "loglog": [0, 1], "sqrt": [0, 1], "k": [0, 1], "model": [0, 1], "axi": [0, 1], "1e": [0, 1], "24": [0, 1], "grid": [0, 1], "rthz": [0, 1], "upper": [0, 1], "_asd": [0, 1], "onli": [0, 1, 3], "between": [0, 1], "properli": [0, 1], "calibr": [0, 1], "That": [0, 1], "ok": [0, 1], "becaus": [0, 1], "high": [0, 1], "sens": [0, 1], "astrophys": [0, 1], "sourc": [0, 1], "12": [0, 1], "captur": [0, 1], "nyquist": [0, 1], "2048": [0, 1], "our": [0, 1], "given": [0, 1], "end": [0, 1, 3], "almost": [0, 1], "alwai": [0, 1], "strong": [0, 1], "instrument": [0, 1], "engin": [0, 1], "mirror": [0, 1], "suspens": [0, 1], "reson": [0, 1], "500": [0, 1], "harmon": [0, 1], "control": [0, 1], "dither": [0, 1], "60": [0, 1], "unwant": [0, 1], "return": [0, 1], "rel": [0, 1], "weak": [0, 1], "less": [0, 1], "than": [0, 1], "long": [0, 1], "while": [0, 1, 3], "entir": [0, 1], "hard": [0, 1], "tune": [0, 1], "ey": [0, 1], "won": [0, 1], "arbitrari": [0, 1], "thing": [0, 1], "much": [0, 1], "accuraci": [0, 1], "metric": [0, 1], "evalu": [0, 1, 3], "sensit": [0, 1], "distanc": [0, 1], "regist": [0, 1], "ratio": [0, 1], "snr": [0, 1], "direct": [0, 1], "orient": [0, 1], "nomin": [0, 1], "threshold": [0, 1], "similar": [0, 1], "cbc": [0, 1], "each": [0, 1], "system": [0, 1], "mass": [0, 1], "sun": [0, 1], "neglig": [0, 1], "spin": [0, 1], "merger": [0, 1], "like": [0, 1, 3], "siren": [0, 1], "theoret": [0, 1], "calcul": [0, 1, 3], "fall": [0, 1], "off": [0, 1], "earth": [0, 1], "tell": [0, 1], "how": [0, 1, 3], "far": [0, 1], "awai": [0, 1], "astronom": [0, 1], "post": [0, 1], "newtonian": [0, 1], "quadrupol": [0, 1], "inspir": [0, 1], "phase": [0, 1], "best": [0, 1], "simpl": [0, 1], "express": [0, 1], "ringdown": [0, 1], "strength": [0, 1], "But": [0, 1], "order": [0, 1, 3], "its": [0, 1, 3], "antenna": [0, 1], "pattern": [0, 1], "respons": [0, 1], "It": [0, 1], "non": [0, 1], "trivial": [0, 1], "2648": [0, 1], "maximum": [0, 1, 3], "valu": [0, 1], "describ": [0, 1, 3], "appendix": [0, 1], "d": [0, 1], "findchirp": [0, 1], "algorithm": [0, 1], "compact": [0, 1], "b": [0, 1], "allen": [0, 1], "et": [0, 1], "al": [0, 1], "physic": [0, 1], "review": [0, 1], "85": [0, 1], "122006": [0, 1], "2012": [0, 1], "arxiv": [0, 1], "ab": [0, 1], "gr": [0, 1], "qc": [0, 1], "0509116": [0, 1], "bns_rang": [0, 1], "spectrum": [0, 1], "f": [0, 1, 3], "copi": [0, 1], "step": [0, 1], "size": [0, 1], "df": [0, 1], "constant": [0, 1], "speed": [0, 1], "light": [0, 1], "clight": [0, 1], "99792458e8": [0, 1], "m": [0, 1], "newton": [0, 1], "67259e": [0, 1], "11": [0, 1, 3], "kg": [0, 1], "parsec": [0, 1], "popular": [0, 1], "26": [0, 1], "year": [0, 1], "08568025e16": [0, 1], "solar": [0, 1], "msol": [0, 1], "989e30": [0, 1], "isn": [0, 1], "fun": [0, 1], "tsol": [0, 1], "background": [0, 1], "snrdet": [0, 1], "convers": [0, 1], "horizon": [0, 1], "favg": [0, 1], "mn": [0, 1], "m1": [0, 1], "m2": [0, 1], "mtot": [0, 1], "total": [0, 1], "eta": [0, 1], "symmetr": [0, 1], "mchirp": [0, 1], "chirp": [0, 1], "follow": [0, 1, 3], "eqn": [0, 1], "1b": [0, 1], "fiduci": [0, 1], "dist": [0, 1], "mpc": [0, 1], "0e6": [0, 1], "innermost": [0, 1], "stabl": [0, 1], "circular": [0, 1], "orbit": [0, 1], "isco": [0, 1], "r_isco": [0, 1], "separ": [0, 1, 3], "geometr": [0, 1], "6m": [0, 1], "pn": [0, 1], "8m": [0, 1], "eob": [0, 1], "f_isco": [0, 1], "pi": [0, 1], "minimum": [0, 1], "too": [0, 1], "ani": [0, 1, 3], "fisco": [0, 1], "fr": [0, 1], "nonzero": [0, 1], "logical_and": [0, 1], "ffr": [0, 1], "stationari": [0, 1], "approx": [0, 1], "htild": [0, 1], "96": [0, 1], "htilda2": [0, 1], "det": [0, 1], "sspec": [0, 1], "els": [0, 1], "sspecfr": [0, 1], "optim": [0, 1], "d2": [0, 1], "d_bn": [0, 1], "r_bn": [0, 1], "1f": [0, 1], "169": [0, 1], "74": [0, 1], "147": [0, 1], "9": [0, 1], "graviti": [0, 1], "thu": [0, 1], "higher": [0, 1], "louder": [0, 1], "veri": [0, 1], "strongli": [0, 1], "color": [0, 1], "fluctuat": [0, 1], "larger": [0, 1], "reach": [0, 1], "roughli": [0, 1], "flat": [0, 1], "white": [0, 1], "80": [0, 1], "300": [0, 1], "divid": [0, 1], "suppress": [0, 1], "extra": [0, 1], "better": [0, 1], "search": [0, 1], "prior": [0, 1], "knowledg": [0, 1], "To": [0, 1, 3], "rid": [0, 1], "remain": [0, 1], "longer": [0, 1], "now": [0, 1, 3], "sigma": [0, 1], "along": [0, 1], "function": [0, 1, 3], "def": 0, "interp_psd": 0, "nt": 0, "rfftfreq": 0, "freqs1": 0, "linspac": 0, "back": [0, 1], "care": [0, 1], "normal": [0, 1, 3], "hf": 0, "rfft": 0, "norm": 0, "white_hf": 0, "white_ht": 0, "irfft": 0, "n": 0, "whiten_data": [0, 1], "strain_h1_whiten": [0, 1], "strain_l1_whiten": [0, 1], "bb": [0, 1], "btype": [0, 1], "strain_h1_whitenbp": [0, 1], "strain_l1_whitenbp": [0, 1], "short": [0, 1], "pick": [0, 1], "shorter": [0, 1], "ftt": [0, 1], "int": [0, 1], "overlap": [0, 1, 3], "resolv": [0, 1], "featur": [0, 1], "novl": [0, 1], "15": [0, 1], "16": [0, 1], "window": [0, 1, 3], "minim": [0, 1, 3], "leakag": [0, 1, 3], "spectral_leakag": [0, 1, 3], "blackman": [0, 1], "colormap": [0, 1], "exampl": [0, 1, 3], "colormaps_refer": [0, 1], "viridi": [0, 1], "seem": [0, 1], "new": [0, 1], "settl": [0, 1], "ocean": [0, 1], "spec_cmap": [0, 1], "spec_h1": [0, 1], "bin": [0, 1], "im": [0, 1], "specgram": [0, 1], "noverlap": [0, 1], "cmap": [0, 1], "xextent": [0, 1], "colorbar": [0, 1], "aligo": [0, 1], "_h1_spectrogram": [0, 1], "_l1_spectrogram": [0, 1], "mai": [0, 1], "excess": [0, 1], "1000": [0, 1], "1500": [0, 1], "evid": [0, 1], "multipl": [0, 1], "violin": [0, 1], "mode": [0, 1], "fiber": [0, 1], "hold": [0, 1], "interferomet": [0, 1], "zoom": [0, 1], "think": [0, 1], "hope": [0, 1], "region": [0, 1], "_h1_spectrogram_whiten": [0, 1], "_l1_spectrogram_whiten": [0, 1], "loud": [0, 1], "visibl": [0, 1, 3], "object": [0, 1], "show": [0, 1], "characterist": [0, 1], "rise": [0, 1], "chang": [0, 1], "variabl": [0, 1], "consist": [0, 1], "parameter": [0, 1], "As": [0, 1], "ident": [0, 1], "re": [0, 1], "skip": [0, 1], "subtleti": [0, 1], "combin": [0, 1], "f_templat": [0, 1], "metadata": [0, 1], "template_p": [0, 1], "template_c": [0, 1], "t_m1": [0, 1], "meta": [0, 1], "attr": [0, 1], "t_m2": [0, 1], "t_a1": [0, 1], "a1": [0, 1], "t_a2": [0, 1], "a2": [0, 1], "t_approx": [0, 1], "close": [0, 1], "extend": [0, 1], "zero": [0, 1], "pad": [0, 1], "length": [0, 1], "template_offset": [0, 1], "template_p_whiten": [0, 1], "template_c_whiten": [0, 1], "template_p_whitenbp": [0, 1], "template_c_whitenbp": [0, 1], "t_mtot": [0, 1], "final": [0, 1], "bh": [0, 1], "95": [0, 1], "initi": [0, 1], "t_mfin": [0, 1], "radiu": [0, 1], "km": [0, 1], "r_fin": [0, 1], "j": [0, 1], "ttime": [0, 1], "instantan": [0, 1], "tphase": [0, 1], "unwrap": [0, 1], "angl": [0, 1], "fgw": [0, 1], "gradient": [0, 1], "fix": [0, 1], "discontinu": [0, 1], "iffix": [0, 1], "100": [0, 1], "001": [0, 1], "v": [0, 1], "c": [0, 1], "voverc": [0, 1], "f_gw": [0, 1], "f_inband": [0, 1], "iband": [0, 1], "peak": [0, 1], "ipeak": [0, 1], "argmax": [0, 1], "cycl": [0, 1], "inband": [0, 1], "ncycl": [0, 1], "famili": [0, 1], "2f": [0, 1], "msun": [0, 1], "mfinal": [0, 1], "durat": [0, 1], "n_cycl": [0, 1], "0f": [0, 1], "subplot": [0, 1], "xlim": [0, 1], "d_eff": [0, 1], "_templat": [0, 1], "gw150914_4_templat": [0, 1], "lalsim": [0, 1], "seobnrv2": [0, 1], "41": [0, 1], "29": [0, 1], "70": [0, 1], "98": [0, 1], "67": [0, 1], "43": [0, 1], "35": [0, 1], "77": [0, 1], "84": [0, 1], "08": [0, 1], "02": [0, 1], "06": [0, 1], "57": [0, 1], "199": [0, 1], "known": [0, 1], "buri": [0, 1], "gaussian": [0, 1], "techniqu": [0, 1], "commun": [0, 1], "noisi": [0, 1], "possibl": [0, 1], "On": [0, 1], "other": [0, 1, 3], "hand": [0, 1], "even": [0, 1, 3], "scientist": [0, 1], "hidden": [0, 1], "compress": [0, 1], "convent": [0, 1], "rather": [0, 1], "elabor": [0, 1], "suit": [0, 1], "against": [0, 1], "procedur": [0, 1], "help": [0, 1], "infer": [0, 1], "sky": [0, 1], "locat": [0, 1], "blind": [0, 1], "250": [0, 1], "000": [0, 1], "trigger": [0, 1], "coincid": [0, 1], "extrem": [0, 1], "computation": [0, 1], "intens": [0, 1], "pipelin": [0, 1], "simplifi": [0, 1], "being": [0, 1], "good": [0, 1], "fairli": [0, 1], "method": [0, 1], "vs": [0, 1], "iv": [0, 1], "coalesc": [0, 1], "1602": [0, 1], "03839": [0, 1], "common": [0, 1, 3], "psd_window": [0, 1], "50": [0, 1, 3], "record": [0, 1], "etim": [0, 1], "datafreq": [0, 1], "fftfreq": [0, 1], "remov": [0, 1], "effect": [0, 1, 3], "window_funct": [0, 1], "tukey_window": [0, 1], "dwindow": [0, 1], "tukei": [0, 1], "alpha": [0, 1], "prefer": [0, 1], "prepar": [0, 1], "template_fft": [0, 1], "data_psd": [0, 1], "data_fft": [0, 1], "power_vec": [0, 1], "interp": [0, 1], "output": [0, 1], "multipli": [0, 1], "space": [0, 1], "invers": [0, 1], "ifft": [0, 1], "put": [0, 1, 3], "conjug": [0, 1], "optimal_tim": [0, 1], "expect": [0, 1], "Then": [0, 1], "sigmasq": [0, 1], "snr_complex": [0, 1], "peaksampl": [0, 1], "roll": [0, 1], "indmax": [0, 1], "timemax": [0, 1], "snrmax": [0, 1], "definit": [0, 1], "d_thresh": [0, 1], "distnac": [0, 1], "offset": [0, 1], "appli": [0, 1], "template_phaseshift": [0, 1], "real": [0, 1], "exp": [0, 1], "1j": [0, 1], "template_rol": [0, 1], "scale": [0, 1], "pass": [0, 1], "template_whiten": [0, 1], "template_match": [0, 1], "4f": [0, 1], "pcolor": [0, 1], "strain_whitenbp": [0, 1], "template_l1": [0, 1], "template_h1": [0, 1], "ylim": 0, "25": 0, "left": [0, 1], "_": 0, "_snr": 0, "h": [0, 1], "stdev": 0, "resid": 0, "residu": 0, "subtract": 0, "_matchtim": 0, "displai": [0, 1], "top": [0, 1], "template_f": 0, "absolut": 0, "_matchfreq": 0, "4395": [0, 1], "814": [0, 1], "1889": [0, 1], "4324": [0, 1], "13": [0, 1], "999": [0, 1], "1650": [0, 1], "bayesian": [0, 1], "posterior": [0, 1], "nearbi": [0, 1], "doe": [0, 1], "job": [0, 1], "uncertain": [0, 1], "somewhat": [0, 1], "Is": [0, 1], "NOT": [0, 1], "actual": [0, 1, 3], "luminos": [0, 1], "depend": [0, 1], "These": [0, 1, 3], "redshift": [0, 1], "cosmolog": [0, 1], "taken": [0, 1], "account": [0, 1], "neglect": [0, 1], "evolut": [0, 1], "been": [0, 1], "themselv": [0, 1], "true": [0, 1], "smaller": [0, 1], "factor": [0, 1], "z": [0, 1], "wav": [0, 1], "downsampl": [0, 1], "2s": [0, 1], "io": 0, "wavfil": 0, "keep": 0, "within": 0, "integ": 0, "write": [0, 1], "write_wavfil": [0, 1, 3], "int16": 0, "32767": 0, "deltat_sound": [0, 1], "indxd": [0, 1], "_h1_whitenbp": [0, 1], "_l1_whitenbp": [0, 1], "template_p_smooth": [0, 1], "soom": [0, 1], "_template_whiten": [0, 1], "With": [0, 1], "headphon": [0, 1], "abl": [0, 1, 3], "hear": [0, 1], "faint": [0, 1], "thump": [0, 1], "middl": [0, 1], "fna": [0, 1], "gw150914_template_whiten": [0, 1], "browser": [0, 1], "support": [0, 1], "element": [0, 1], "gw150914_h1_whitenbp": [0, 1], "enhanc": [0, 1], "increas": [0, 1], "nasa": [0, 1], "emploi": [0, 1], "telescop": [0, 1], "imag": [0, 1], "fals": [0, 1], "400": [0, 1], "ing": [0, 1], "notic": [0, 1], "pitch": [0, 1], "easier": [0, 1], "reqshift": [0, 1, 3], "fshift": [0, 1], "sample_r": [0, 1], "x": 0, "float": [0, 1], "nbin": 0, "shape": 0, "y": 0, "speedup": [0, 1], "fss": [0, 1], "strain_h1_shift": [0, 1], "strain_l1_shift": [0, 1], "_h1_shift": [0, 1], "_l1_shift": [0, 1], "template_p_shift": [0, 1], "_template_shift": [0, 1], "gw150914_template_shift": [0, 1], "gw150914_h1_shift": [0, 1], "mention": [0, 1], "introduct": [0, 1], "check": [0, 1, 3], "repeat": [0, 1], "16384": [0, 1], "relev": [0, 1], "dq": [0, 1], "hw": [0, 1], "inject": [0, 1], "data_seg": [0, 1], "fn": [0, 1], "l": [0, 1], "l1_losc_4_v1": [0, 1], "kei": [0, 1], "pair": [0, 1], "item": [0, 1, 3], "isnan": [0, 1], "start": [0, 1, 3], "stop": [0, 1], "level": [0, 1], "cbc_cat3": [0, 1], "conserv": [0, 1], "choic": [0, 1], "dqflag": [0, 1], "continu": [0, 1, 3], "segment_list": [0, 1], "dq_channel_to_seglist": [0, 1], "iseg": [0, 1], "time_seg": [0, 1], "seg_strain": [0, 1], "would": [0, 1], "insert": [0, 1], "hardwar": [0, 1], "no_cbc_hw_inj": [0, 1], "cbc_cat1": [0, 1], "cbc_cat2": [0, 1], "burst_cat1": [0, 1], "burst_cat2": [0, 1], "burst_cat3": [0, 1], "no_burst_hw_inj": [0, 1], "no_detchar_hw_inj": [0, 1], "no_cw_hw_inj": [0, 1], "no_stoch_hw_inj": [0, 1], "acquir": [0, 1], "save": [0, 1, 3], "disk": [0, 1], "memori": [0, 1], "suffici": [0, 1], "f_nyquist": [0, 1], "equal": [0, 1], "spinless": [0, 1], "1557": [0, 1], "m_tot": [0, 1], "canon": [0, 1], "howev": [0, 1], "interest": [0, 1], "irang": [0, 1], "structur": [0, 1], "dat": [0, 1], "datcsv": [0, 1], "arrai": [0, 1], "transpos": [0, 1], "header": [0, 1], "fncsv": [0, 1], "_data": [0, 1], "headcsv": [0, 1], "h1_data_whiten": [0, 1], "l1_data_whiten": [0, 1], "h1_template_whiten": [0, 1], "l1_template_whiten": [0, 1], "fmtcsv": [0, 1], "join": [0, 1], "6f": [0, 1], "savetxt": [0, 1], "fmt": [0, 1], "wrote": [0, 1], "click": [0, 1], "jupyt": [0, 1], "corner": [0, 1], "menu": [0, 1], "azur": [0, 1], "gw150914_data": [0, 1], "util": 1, "ul": 1, "h1_losc_4_v2": 1, "l1_losc_4_v2": 1, "utcev": 1, "2015": 1, "09": 1, "14t09": 1, "45": 1, "743": 1, "237": 1, "355": 1, "769": 1, "0428999418774637e": 1, "sigplot": 1, "statist": 3, "due": 3, "fridai": 3, "04": 3, "59pm": 3, "pt": 3, "prof": 3, "p\u00e9rez": 3, "gsi": 3, "sapienza": 3, "depart": 3, "thi": 3, "worth": 3, "type": 3, "group": 3, "For": 3, "conclus": 3, "hw02": 3, "us": 3, "again": 3, "code": 3, "ligo": 3, "gravit": 3, "wave": 3, "detect": 3, "tutori": 3, "ar": 3, "solut": 3, "document": 3, "complet": 3, "pictur": 3, "The": 3, "data": 3, "layout": 3, "reorgan": 3, "still": 3, "file": 3, "live": 3, "audio": 3, "goe": 3, "present": 3, "befor": 3, "git": 3, "empti": 3, "littl": 3, "hack": 3, "gitkeep": 3, "those": 3, "explain": 3, "here": 3, "hw05": 3, "made": 3, "local": 3, "could": 3, "current": 3, "discuss": 3, "appropri": 3, "pyproj": 3, "toml": 3, "cfg": 3, "mytoi": 3, "guid": 3, "next": 3, "note": 3, "list": 3, "author": 3, "scientif": 3, "collabor": 3, "lsc": 3, "name": 3, "four": 3, "small": 3, "subfold": 3, "tests_readligo": 3, "command": 3, "pytest": 3, "modul": 3, "move": 3, "whiten": 3, "updat": 3, "instead": 3, "find": 3, "choos": 3, "spectral": 3, "http": 3, "org": 3, "three": 3, "set": 3, "proper": 3, "build": 3, "main": 3, "rememb": 3, "hub": 3, "view": 3, "book": 3, "vnc": 3, "desktop": 3, "sphinx": 3, "manual": 3, "instruct": 3, "do": 3, "lectur": 3, "url": 3, "push": 3, "branch": 3, "repo": 3, "talk": 3, "respect": 3, "lab": 3, "session": 3, "target": 3, "env": 3, "environ": 3, "clone": 3, "clean": 3, "_build": 3, "folder": 3, "overal": 3, "workflow": 3, "Be": 3, "sure": 3, "hw04": 3, "groupxx": 3, "xx": 3, "clear": 3, "commit": 3, "progress": 3, "tag": 3, "previou": 3, "readm": 3, "md": 3, "descript": 3, "project": 3, "badg": 3, "directli": 3, "launch": 3, "Not": 3, "archiv": 3, "gitignor": 3, "contribut": 3, "statement": 3}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"binari": [0, 1], "black": [0, 1], "hole": [0, 1], "signal": [0, 1], "IN": [0, 1], "ligo": [0, 1], "open": [0, 1], "data": [0, 1], "thi": [0, 1], "tutori": [0, 1], "intend": [0, 1], "educ": [0, 1], "purpos": [0, 1], "The": [0, 1], "code": [0, 1], "shown": [0, 1], "here": [0, 1], "us": [0, 1], "produc": [0, 1], "result": [0, 1], "paper": [0, 1], "publish": [0, 1], "scientif": [0, 1], "collabor": [0, 1], "which": [0, 1], "instead": [0, 1], "reli": [0, 1], "special": [0, 1], "analysi": [0, 1], "softwar": [0, 1], "packag": [0, 1, 3], "For": [0, 1], "publicli": [0, 1], "avail": [0, 1], "gravit": [0, 1], "wave": [0, 1], "ar": [0, 1], "lsc": [0, 1], "virgo": [0, 1], "see": [0, 1], "http": [0, 1], "www": [0, 1], "gwosc": [0, 1], "org": [0, 1], "technic": [0, 1], "note": [0, 1], "bbh_tutorial_not": [0, 1], "tabl": [0, 1], "content": [0, 1], "intro": [0, 1], "process": [0, 1], "download": [0, 1], "comput": [0, 1], "python": [0, 1], "instal": [0, 1, 3], "set": [0, 1], "event": [0, 1], "name": [0, 1], "choos": [0, 1], "plot": [0, 1, 3], "type": [0, 1], "read": [0, 1], "properti": [0, 1], "from": [0, 1, 3], "local": [0, 1], "json": [0, 1], "file": [0, 1], "advanc": [0, 1], "gap": [0, 1], "first": [0, 1], "look": [0, 1], "h1": [0, 1], "l1": [0, 1], "amplitud": [0, 1], "spectral": [0, 1], "densiti": [0, 1], "asd": [0, 1], "neutron": [0, 1], "star": [0, 1], "bn": [0, 1], "detect": [0, 1], "rang": [0, 1], "bbh": [0, 1], "whiten": [0, 1], "spectrogram": [0, 1], "waveform": [0, 1], "templat": [0, 1], "match": [0, 1], "filter": [0, 1], "find": [0, 1], "make": [0, 1, 3], "sound": [0, 1], "listen": [0, 1], "frequenc": [0, 1], "shift": [0, 1], "audio": [0, 1], "segment": [0, 1], "comment": [0, 1], "sampl": [0, 1], "rate": [0, 1], "construct": [0, 1], "csv": [0, 1], "contain": [0, 1], "homework": 3, "No": 3, "4": 3, "notebook": 3, "research": 3, "part": 3, "ii": 3, "deliver": 3, "5": 3, "point": 3, "repositori": 3, "structur": 3, "add": 3, "test": 3, "readligo": 3, "py": 3, "creat": 3, "util": 3, "new": 3, "jupyterbook": 3, "github": 3, "page": 3, "action": 3, "makefil": 3}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinxcontrib.bibtex": 9, "sphinx": 56}})
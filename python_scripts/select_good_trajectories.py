def define_criteria_good_tracks_default(crStats):

    crStats.criteria['isolate_tracks_good'] = \
        ~crStats.criteria['isolate_tracks_bad_framerate'] \
        & crStats.criteria['isolate_tracks_complete']

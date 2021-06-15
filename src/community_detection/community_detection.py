
import logging
import src.community_detection.girvan_newman as girvan_newman
import src.community_detection.k_means as k_means
import src.community_detection.label_propagation as label_propagation


logger = logging.getLogger('sna')


def run(graph, args):
    number_of_communities = 0

    if args.community_method == 'girvan-newman':
        logger.info(f'\nCommunity detection procedure: Girvan-Newman')
        communities, number_of_communities = girvan_newman.detect(graph, args)

    if args.community_method == 'label-propagation':
        logger.info(f'\nCommunity detection procedure: Label propagation')
        communities, number_of_communities = label_propagation.detect(graph, args)

    if number_of_communities > 0:
        k_means.detect(graph, number_of_communities, args)

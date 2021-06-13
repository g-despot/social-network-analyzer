
import logging
import src.community_detection.girvan_newman as girvan_newman
import src.community_detection.label_propagation as label_propagation


logger = logging.getLogger('sna')


def run(graph, args):
    if args.method == 'girvan-newman':
        logger.info(f'\nCommunity detection procedure: Girvan-Newman')
        girvan_newman.detect(graph, args)

    if args.method == 'label-propagation':
        logger.info(f'\nCommunity detection procedure: Label propagation')
        label_propagation.detect(graph, args)

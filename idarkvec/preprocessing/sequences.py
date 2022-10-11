
class SequenceExtractor():
    
    @staticmethod
    def _extract_by_ports(df, top_ports):
        df['pp'] = df['dst_port'].astype(str)+'/'+df['proto'].astype(str)
        topN = df.value_counts('pp').iloc[:top_ports].index
        df.loc[df[~df.pp.isin(topN)].index, 'pp'] = 'other'
        # Extract IPs sequences by ports
        sequences = df.sort_values('ts').groupby('pp')\
                                             .agg({'src_ip':list}).sort_index()
        
        return sequences